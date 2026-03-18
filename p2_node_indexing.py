"""
node registry and reaction diff for SIMG
---------------
  ("Atom", map_id)                        — atom node
  ("LP",   map_id, slot)                  — lone pair node
  ("BO",   mi, mj, slot)                  — bond orbital node  (mi <= mj)
  ("A2B",  map_id, bo_mi, bo_mj, bo_slot) — atom → bond-orbital directed edge
  ("INT",  key_A,  key_B)                 — orbital interaction  (key_A < key_B)

Slot ordering
-------------
  LP  : primary   — (p - s) ascending     → slot 0 = s-rich, slot N = p-rich/conjugated
        secondary — occupancy descending   → stable tiebreak when p-s values are equal
        Rationale: mirrors SIMG paper conjugation criterion (p - s > 80 → conjugated LP)

  BO  : occupancy descending               → slot 0 = σ bonding (~2.0e),
                                              slot 1 = π bonding (double/triple bonds only)
        Rationale: bond order directly encodes σ/π distinction;
                   no antibonding nodes exist in the SIMG prediction graph.

Feature layout in node_preds  (N_total × 16 tensor, columns 0-indexed)
-----------------------------------------------------------------------
  Atom  cols  0- 3 : Charge, Core electrons, Valence electrons, Total electrons
  LP    cols  4- 8 : s character, p character, d character, f character, occupancy
  BO    cols  9-15 : occupancy, s diff, p diff, d diff, f diff,
                     polarisation diff, polarisation coeff diff

Feature layout in a2b_preds  (E_a2b × 6 tensor)
------------------------------------------------
  cols  0- 5 : s, p, d, f (atom's hybridisation contribution to this BO),
               polarisation, polarisation coefficient  — all normalised to [0,1]

Feature layout in int_preds  (E_int × 3 tensor)
------------------------------------------------
  cols  0- 2 : perturbation energy (E2 / 100), energy difference (ΔE),
               Fock matrix element
"""

import torch
from collections import defaultdict
from typing import Dict, Tuple, Any

# ── column indices into node_preds ────────────────────────────────────────────
_ATOM_COLS  = slice(0, 4)    # Charge, Core, Valence, Total
_LP_COLS    = slice(4, 9)    # s, p, d, f, occupancy
_BO_COLS    = slice(9, 16)   # occupancy, s, p, d, f, pol_diff, pol_coeff_diff

_LP_S_COL   = 4
_LP_P_COL   = 5
_LP_OCC_COL = 8   # secondary sort key

_BO_OCC_COL = 9

# ── per-node-type masks: zero out irrelevant columns before storing ───────────
# All node_preds rows are 16-d regardless of node type. Only the columns
# meaningful for each type are kept; the rest are zeroed so the stored
# tensor is clean for downstream use (e.g. stacking into FM input matrix).
_NODE_PREDS_DIM = 16
_ATOM_MASK = torch.zeros(_NODE_PREDS_DIM, dtype=torch.bool)
_ATOM_MASK[0:4]  = True   # Charge, Core, Valence, Total
_LP_MASK   = torch.zeros(_NODE_PREDS_DIM, dtype=torch.bool)
_LP_MASK[4:9]    = True   # s, p, d, f, occupancy
_BO_MASK   = torch.zeros(_NODE_PREDS_DIM, dtype=torch.bool)
_BO_MASK[9:16]   = True   # occupancy, s/p/d/f diff, pol_diff, pol_coeff_diff

# ── feature name maps (for get_node_features) ────────────────────────────────
_ATOM_FEAT_NAMES = ["charge", "core_electrons", "valence_electrons", "total_electrons"]
_LP_FEAT_NAMES   = ["s_char", "p_char", "d_char", "f_char", "occupancy"]
_BO_FEAT_NAMES   = ["occupancy", "s_diff", "p_diff", "d_diff", "f_diff",
                    "pol_diff", "pol_coeff_diff"]
_A2B_FEAT_NAMES  = ["s_contrib", "p_contrib", "d_contrib", "f_contrib",
                    "polarisation", "pol_coeff"]
_INT_FEAT_NAMES  = ["perturbation_energy", "energy_diff", "fock_element"]

NodeKey = Tuple


# ════════════════════════════════════════════════════════════════════════════
# Public helper — named feature access
# ════════════════════════════════════════════════════════════════════════════

def get_node_features(key: NodeKey, feat_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Convert a raw feature tensor to a named dict based on the node type in key.

    Parameters
    ----------
    key        : one of the NodeKey tuples produced by index_all_nodes()
    feat_tensor: the tensor stored in the registry for that key

    Returns
    -------
    dict[str, float]  e.g. {"occupancy": 1.97, "s_diff": 0.02, ...}
    """
    # Tensors in the registry are already masked (irrelevant cols = 0).
    # We still slice to the meaningful range so the dict contains only
    # the named values (no trailing zeros).
    t = feat_tensor.detach().cpu().float()
    node_type = key[0]

    if node_type == "Atom":
        return dict(zip(_ATOM_FEAT_NAMES, t[0:4].tolist()))
    elif node_type == "LP":
        return dict(zip(_LP_FEAT_NAMES,   t[4:9].tolist()))
    elif node_type == "BO":
        return dict(zip(_BO_FEAT_NAMES,   t[9:16].tolist()))
    elif node_type == "A2B":
        return dict(zip(_A2B_FEAT_NAMES,  t[:6].tolist()))
    elif node_type == "INT":
        return dict(zip(_INT_FEAT_NAMES,  t[:3].tolist()))
    else:
        raise ValueError(f"Unknown node type: {node_type}")


# ════════════════════════════════════════════════════════════════════════════
# Section 1 — LP ownership (positional, no edge scanning)
# ════════════════════════════════════════════════════════════════════════════

def _lp_owner_atoms(atoms_per_lp: list, n_atoms: int) -> Dict[int, int]:
    """
    global LP node index  →  owning atom's RDKit index.

    atoms_per_lp[k] is the RDKit atom index that owns LP_k,
    produced by get_atoms_per_lp(n_lps_per_atom) in model_utils.py.
    """
    return {n_atoms + k: rdkit_i for k, rdkit_i in enumerate(atoms_per_lp)}


# ════════════════════════════════════════════════════════════════════════════
# Section 2 — BO ownership (via a2b_index)
# ════════════════════════════════════════════════════════════════════════════

def _bo_owner_pairs(data, mol_map_ids: list, drop_h: bool
                    ) -> Dict[int, Tuple[int, int]]:
    """
    BO node global index  →  (map_i, map_j)  with map_i <= map_j.
    """
    n_atoms = int(data.is_atom.sum())
    a2b     = data.a2b_index
    assert a2b.shape[0] == 2, f"a2b_index shape must be (2, E), got {a2b.shape}"

    bondnode2atoms: Dict[int, list] = defaultdict(list)
    for atom_idx, bond_idx in zip(a2b[0].tolist(), a2b[1].tolist()):
        bondnode2atoms[bond_idx].append(atom_idx)

    bo_owner: Dict[int, Tuple[int, int]] = {}
    for bond_node_idx, atom_list in bondnode2atoms.items():
        if len(atom_list) != 2:
            continue                              # incomplete — skip
        a0, a1 = atom_list
        if a0 >= n_atoms or a1 >= n_atoms:
            continue                              # non-atom endpoint — skip
        mi = int(mol_map_ids[a0])
        mj = int(mol_map_ids[a1])
        if drop_h and (mi == 0 or mj == 0):
            continue
        bo_owner[bond_node_idx] = (mi, mj) if mi <= mj else (mj, mi)

    return bo_owner


# ════════════════════════════════════════════════════════════════════════════
# Section 3 — index_all_nodes  (core function)
# ════════════════════════════════════════════════════════════════════════════

def index_all_nodes(
        run_out: Dict[str, Any],
        drop_h: bool = True,
) -> Dict[NodeKey, torch.Tensor]:
    """
    Build a unified node/edge registry for one molecule.
      ("Atom", map_id)                  → node_preds row  (16-d, cols 0-3 active,   rest zero)
      ("LP",   map_id, slot)            → node_preds row  (16-d, cols 4-8 active,   rest zero)
      ("BO",   mi, mj, slot)            → node_preds row  (16-d, cols 9-15 active,  rest zero)
      ("A2B",  map_id, mi, mj, slot)    → a2b_preds  row  ( 6-d)
      ("INT",  key_A,  key_B)           → int_preds  row  ( 3-d)
                                          (averaged if both directions present)

    Parameters
    ----------
    run_out : output dict from simg
    drop_h
    """
    data        = run_out["data"]
    mol_map_ids = run_out["mol_map_ids"]
    node_preds  = torch.as_tensor(run_out["node_preds"]).float()
    a2b_preds   = run_out["a2b_preds"]
    int_preds   = run_out["int_preds"]

    registry: Dict[NodeKey, torch.Tensor] = {}

    is_atom = data.is_atom.bool()
    is_lp   = data.is_lp.bool()
    is_bond = data.is_bond.bool()

    n_atoms = int(is_atom.sum())

    atom_node_indices = torch.nonzero(is_atom).view(-1).tolist()
    lp_node_indices   = torch.nonzero(is_lp).view(-1).tolist()

    assert len(atom_node_indices) == len(mol_map_ids), (
        f"Atom count mismatch: {len(atom_node_indices)} graph nodes "
        f"vs {len(mol_map_ids)} map ids"
    )

    # ── Node type 1: Atoms ───────────────────────────────────────────────────
    for rdkit_i, node_idx in enumerate(atom_node_indices):
        mid = int(mol_map_ids[rdkit_i])
        if mid <= 0:
            continue                  # always skip unmapped atoms (H)
        v = node_preds[node_idx].clone()
        v[~_ATOM_MASK] = 0.0
        registry[("Atom", mid)] = v

    # ── Node type 2: Lone Pairs ──────────────────────────────────────────────
    # Slot ordering: primary = (p - s) ascending  (s-rich → p-rich/conjugated)
    #                secondary = occupancy descending  (tiebreak)
    atoms_per_lp = run_out["atoms_per_lp"]
    lp_owner     = _lp_owner_atoms(atoms_per_lp, n_atoms)

    atom_lp_groups: Dict[int, list] = defaultdict(list)
    for lp_node_idx in lp_node_indices:
        rdkit_atom_idx = lp_owner.get(lp_node_idx)
        if rdkit_atom_idx is None:
            continue
        mid = int(mol_map_ids[rdkit_atom_idx])
        if mid <= 0 and drop_h:
            continue
        atom_lp_groups[mid].append(lp_node_idx)

    for mid, lp_nodes in atom_lp_groups.items():
        lp_nodes_sorted = sorted(
            lp_nodes,
            key=lambda idx: (
                float(node_preds[idx, _LP_P_COL] - node_preds[idx, _LP_S_COL]),   # primary
                -float(node_preds[idx, _LP_OCC_COL]),                              # tiebreak
            )
        )
        for slot, lp_node_idx in enumerate(lp_nodes_sorted):
            v = node_preds[lp_node_idx].clone()
            v[~_LP_MASK] = 0.0
            registry[("LP", mid, slot)] = v

    # ── Node type 3: Bond Orbitals ───────────────────────────────────────────
    # Slot ordering: occupancy descending → slot 0 = σ bonding, slot 1 = π (if present)
    bo_owner = _bo_owner_pairs(data, mol_map_ids, drop_h=drop_h)

    pair_bo_groups: Dict[Tuple[int, int], list] = defaultdict(list)
    for bo_node_idx, pair in bo_owner.items():
        pair_bo_groups[pair].append(bo_node_idx)

    # pre-build reverse map for A2B and INT sections
    bo_node_to_key: Dict[int, NodeKey] = {}

    for pair, bo_nodes in pair_bo_groups.items():
        bo_nodes_sorted = sorted(
            bo_nodes,
            key=lambda idx: float(node_preds[idx, _BO_OCC_COL]),
            reverse=True,
        )
        mi, mj = pair
        for slot, bo_node_idx in enumerate(bo_nodes_sorted):
            key = ("BO", mi, mj, slot)
            v = node_preds[bo_node_idx].clone()
            v[~_BO_MASK] = 0.0
            registry[key] = v
            bo_node_to_key[bo_node_idx] = key

    # ── Node type 4: Atom-Bond edges (A2B) ──────────────────────────────────
    # Key encodes which atom participates and which specific BO it connects to.
    # The BO slot is inherited from the BO key — no independent slot needed.
    if a2b_preds is not None:
        a2b_preds_t = torch.as_tensor(a2b_preds).float()
        a2b_index   = data.a2b_index
        assert a2b_index.shape[0] == 2

        for edge_i, (atom_node_idx, bo_node_idx) in enumerate(
            zip(a2b_index[0].tolist(), a2b_index[1].tolist())
        ):
            if atom_node_idx >= n_atoms:
                continue                        # sanity: src must be an atom
            rdkit_i = atom_node_indices.index(atom_node_idx)
            mid = int(mol_map_ids[rdkit_i])
            if mid <= 0 and drop_h:
                continue
            bo_key = bo_node_to_key.get(bo_node_idx)
            if bo_key is None:
                continue                        # BO was filtered (e.g. H bond)
            _, bo_mi, bo_mj, bo_slot = bo_key
            registry[("A2B", mid, bo_mi, bo_mj, bo_slot)] = a2b_preds_t[edge_i]

    # ── Node type 5: Orbital Interactions (INT) ──────────────────────────────
    # Both LP and BO nodes can be INT endpoints.
    # Canonical key ordering (key_A < key_B) collapses directed pairs.
    # If both directions are predicted, features are averaged.
    if int_preds is not None:
        int_preds_t  = torch.as_tensor(int_preds).float()
        int_edge_idx = data.interaction_edge_index   # (2, E_int)

        if int_edge_idx.shape[1] > 0:
            # build full orbital → key map (LP + BO)
            orbital_to_key: Dict[int, NodeKey] = {}

            for mid, lp_nodes in atom_lp_groups.items():
                lp_sorted = sorted(
                    lp_nodes,
                    key=lambda idx: (
                        float(node_preds[idx, _LP_P_COL] - node_preds[idx, _LP_S_COL]),
                        -float(node_preds[idx, _LP_OCC_COL]),
                    )
                )
                for slot, lp_node_idx in enumerate(lp_sorted):
                    orbital_to_key[lp_node_idx] = ("LP", mid, slot)

            orbital_to_key.update(bo_node_to_key)   # BO keys already built above

            for edge_i, (src, dst) in enumerate(
                zip(int_edge_idx[0].tolist(), int_edge_idx[1].tolist())
            ):
                key_A = orbital_to_key.get(src)
                key_B = orbital_to_key.get(dst)
                if key_A is None or key_B is None:
                    continue
                if key_A > key_B:
                    key_A, key_B = key_B, key_A      # canonical order
                int_key = ("INT", key_A, key_B)
                if int_key in registry:
                    registry[int_key] = (registry[int_key] + int_preds_t[edge_i]) / 2
                else:
                    registry[int_key] = int_preds_t[edge_i]

    return registry


# ════════════════════════════════════════════════════════════════════════════
# Section 4 — reaction diff
# ════════════════════════════════════════════════════════════════════════════

def diff_reaction_nodes(
        registry_r: Dict[NodeKey, torch.Tensor],
        registry_p: Dict[NodeKey, torch.Tensor],
) -> Tuple[Dict, Dict, Dict]:
    """
    Compare reactant and product registries.
    Returns
    -------
    shared    : {key: {"feat_r", "feat_p", "delta"}}
                Keys present on both sides.  delta = feat_p - feat_r.
    only_in_r : {key: tensor}   — broken bonds, lost LPs, leaving atoms
    only_in_p : {key: tensor}   — formed bonds, gained LPs, arriving atoms
    """
    keys_r = set(registry_r)
    keys_p = set(registry_p)

    shared: Dict[NodeKey, Dict] = {
        k: {
            "feat_r": registry_r[k],
            "feat_p": registry_p[k],
            "delta":  registry_p[k] - registry_r[k],
        }
        for k in keys_r & keys_p
    }
    only_in_r = {k: registry_r[k] for k in keys_r - keys_p}
    only_in_p = {k: registry_p[k] for k in keys_p - keys_r}

    return shared, only_in_r, only_in_p


# ════════════════════════════════════════════════════════════════════════════
# Section 5 — end-to-end reaction wrapper
# ════════════════════════════════════════════════════════════════════════════

def index_reaction(
        mapped_rxn_smiles: str,
        simg_fn,
        lp_model,
        gnn_model,
        threshold: float = 0.0,
        use_threshold: bool = False,
        drop_h: bool = True,
        seed: int = 0,
        cache_dir: str = None,
):
    """
    Full pipeline: mapped reaction SMILES → registries + diff.

    Parameters
    ----------
    mapped_rxn_smiles
    simg_fn
    drop_h            : drop nodes/edges involving unmapped H atoms (default True,
                        recommended — H atoms have no cross-reaction identity)

    Returns
    -------
    registry_r, registry_p, shared, only_in_r, only_in_p
    """
    from rdkit import Chem

    def _side_registry(smiles_list):
        merged = {}
        for smi in smiles_list:
            if cache_dir is not None:
                # Cross-session determinism: load pre-generated conformer.
                # The mol already has 3D coords so SIMG_f skips ETKDGv3.
                from e2 import load_conformer
                mol = load_conformer(smi, cache_dir)
            else:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {smi!r}")
                mol = Chem.AddHs(mol)
            run_out = simg_fn(
                mol, lp_model, gnn_model,
                threshold=threshold,
                use_threshold=use_threshold,
                seed=seed,
            )
            merged.update(index_all_nodes(run_out, drop_h=drop_h))
        return merged

    left, right  = mapped_rxn_smiles.split(">>")
    registry_r   = _side_registry([s for s in left.split(".")  if s])
    registry_p   = _side_registry([s for s in right.split(".") if s])
    shared, only_in_r, only_in_p = diff_reaction_nodes(registry_r, registry_p)

    return registry_r, registry_p, shared, only_in_r, only_in_p