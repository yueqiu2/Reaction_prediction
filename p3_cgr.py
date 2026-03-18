"""
pyg HeteroData
CGR structure
---------------
Node types:
    "atom"  — heavy atoms (map_id > 0)
    "lp"    — lone pair orbitals
    "bo"    — bond orbitals (σ / π)

Edge types:
    ("atom", "bond",     "atom")   — covalent bonds from CGR (union of R + P)
    ("atom", "owns_lp",  "lp" )   — atom owns this lone pair (LP ownership)
    ("lp",   "owned_by", "atom")  — reverse of above
    ("atom", "a2b",      "bo" )   — atom contributes to bond orbital
    ("bo",   "a2b_rev",  "atom")  — reverse of above
    ("lp",   "int",      "bo" )   — orbital interaction LP → BO
    ("bo",   "int",      "lp" )   — reverse
    ("lp",   "int",      "lp" )   — orbital interaction LP → LP
    ("bo",   "int",      "bo" )   — orbital interaction BO → BO

Node features
-------------
    atom : [simg_4d  | rxn_flag_3d | delta_4d ]   = 11-d
    lp   : [simg_5d  | rxn_flag_3d | delta_5d ]   = 13-d
    bo   : [simg_7d  | rxn_flag_3d | delta_7d ]   = 17-d

    rxn_flag : one-hot [only_r, only_p, shared]
    delta    : feat_p - feat_r for shared nodes, zeros for only_r / only_p

Edge features
-------------
    atom-atom bond : [bond_type_4d | rxn_flag_3d]  = 7-d
    atom-lp        : no features (topology only)
    a2b            : [a2b_6d]                       = 6-d
    int            : [int_3d]                       = 3-d

Bond types (one-hot, 4-d): single / double / triple / aromatic
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any, List
from rdkit import Chem
from torch_geometric.data import HeteroData

# reaction flag one-hots
_FLAG_ONLY_R  = torch.tensor([1., 0., 0.])   # broken / lost
_FLAG_ONLY_P  = torch.tensor([0., 1., 0.])   # formed / gained
_FLAG_SHARED  = torch.tensor([0., 0., 1.])   # unchanged or changed

# bond type one-hots (single / double / triple / aromatic)
_BOND_SINGLE   = torch.tensor([1., 0., 0., 0.])
_BOND_DOUBLE   = torch.tensor([0., 1., 0., 0.])
_BOND_TRIPLE   = torch.tensor([0., 0., 1., 0.])
_BOND_AROMATIC = torch.tensor([0., 0., 0., 1.])

# feature dimensions
_ATOM_SIMG_DIM = 4    # cols 0-3
_LP_SIMG_DIM   = 5    # cols 4-8
_BO_SIMG_DIM   = 7    # cols 9-15
_A2B_DIM       = 6
_INT_DIM       = 3
_FLAG_DIM      = 3
_BOND_TYPE_DIM = 4

ATOM_FEAT_DIM  = _ATOM_SIMG_DIM + _FLAG_DIM + _ATOM_SIMG_DIM   # 11
LP_FEAT_DIM    = _LP_SIMG_DIM   + _FLAG_DIM + _LP_SIMG_DIM     # 13
BO_FEAT_DIM    = _BO_SIMG_DIM   + _FLAG_DIM + _BO_SIMG_DIM     # 17
BOND_EDGE_DIM  = _BOND_TYPE_DIM + _FLAG_DIM                     # 7
A2B_EDGE_DIM   = _A2B_DIM                                       # 6
INT_EDGE_DIM   = _INT_DIM                                        # 3


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════

def _simg_feat(key, registry):
    """
    Return the SIMG feature slice for a key from a registry.
    feature indexing accords to graph_construction.yaml
    """
    t = registry[key].float()
    if key[0] == "Atom":
        return t[0:4]
    elif key[0] == "LP":
        return t[4:9]
    elif key[0] == "BO":
        return t[9:16]
    raise ValueError(f"Unknown node type: {key[0]}")


def _make_node_feat(key, shared, only_r, only_p):
    """
    Build full node feature vector:
        [simg_feats | rxn_flag | delta]

    For only_r / only_p nodes, delta is 0
    For shared nodes, delta = feat_p - feat_r
    """
    ntype = key[0]
    dim = {"Atom": _ATOM_SIMG_DIM, "LP": _LP_SIMG_DIM, "BO": _BO_SIMG_DIM}[ntype]

    if key in only_r:
        simg  = _simg_feat(key, only_r)
        flag  = _FLAG_ONLY_R
        delta = torch.zeros(dim)
    elif key in only_p:
        simg  = _simg_feat(key, only_p)
        flag  = _FLAG_ONLY_P
        delta = torch.zeros(dim)
    else:   # shared
        entry = shared[key]
        simg  = _simg_feat(key, {key: entry["feat_r"]})
        flag  = _FLAG_SHARED
        delta = (entry["feat_p"] - entry["feat_r"])[
            {"Atom": slice(0,4), "LP": slice(4,9), "BO": slice(9,16)}[ntype]
        ].float()

    return torch.cat([simg, flag, delta])   # shape: (dim*2 + 3,)


def _bond_type_vec(bond):
    """RDKit bond → 4-d one-hot."""
    if bond.GetIsAromatic():
        return _BOND_AROMATIC
    bt = bond.GetBondTypeAsDouble()
    if bt == 1.0:   return _BOND_SINGLE
    if bt == 2.0:   return _BOND_DOUBLE
    if bt == 3.0:   return _BOND_TRIPLE
    return _BOND_SINGLE   # fallback


def _get_bond_change_flag(mi, mj, shared, only_r, only_p):
    """
    Determine whether a bond (mi, mj) is broken, formed, or unchanged,
    based on the BO nodes associated with this atom pair.
    Returns the most significant flag — broken > formed > shared.
    """
    # Check slot 0 (σ bond always exists if any bond exists)
    key_r = ("BO", min(mi,mj), max(mi,mj), 0)
    if key_r in only_r:
        return _FLAG_ONLY_R
    if key_r in only_p:
        return _FLAG_ONLY_P
    return _FLAG_SHARED


def _split_rxn(mapped_rxn_smiles):
    """Split reaction SMILES -> RDKit mols."""
    left, right = mapped_rxn_smiles.split(">>")

    # Combine all reactant fragments into one mol
    rct_parts = [Chem.MolFromSmiles(s) for s in left.split(".") if s]
    pdt_parts = [Chem.MolFromSmiles(s) for s in right.split(".") if s]

    rct = rct_parts[0]
    for m in rct_parts[1:]:
        rct = Chem.CombineMols(rct, m)

    pdt = pdt_parts[0]
    for m in pdt_parts[1:]:
        pdt = Chem.CombineMols(pdt, m)

    return rct, pdt


def _cgr_bonds(rct_mol, pdt_mol):
    """
    Build CGR bond list: both reactant and product bonds, (min_map, max_map).
    Returns list: [(mi, mj, bond_type_vec, flag_vec)]
    """
    def mol_bonds(mol):
        bonds = {}
        for bond in mol.GetBonds():
            mi = bond.GetBeginAtom().GetAtomMapNum()
            mj = bond.GetEndAtom().GetAtomMapNum()
            if mi == 0 or mj == 0:
                continue   # skip unmapped (H) bonds
            key = (min(mi,mj), max(mi,mj))
            bonds[key] = bond
        return bonds

    r_bonds = mol_bonds(rct_mol)
    p_bonds = mol_bonds(pdt_mol)
    all_keys = set(r_bonds) | set(p_bonds)

    result = []
    for (mi, mj) in sorted(all_keys):
        if (mi, mj) in r_bonds and (mi, mj) in p_bonds:
            flag  = _FLAG_SHARED
            btype = _bond_type_vec(r_bonds[(mi,mj)])   # use reactant type
        elif (mi, mj) in r_bonds:
            flag  = _FLAG_ONLY_R
            btype = _bond_type_vec(r_bonds[(mi,mj)])
        else:
            flag  = _FLAG_ONLY_P
            btype = _bond_type_vec(p_bonds[(mi,mj)])
        result.append((mi, mj, btype, flag))

    return result


# ════════════════════════════════════════════════════════════════════════════
# Main public function
# ════════════════════════════════════════════════════════════════════════════

def build_cgr(
        mapped_rxn_smiles: str,
        reg_r: Dict,
        reg_p: Dict,
        shared: Dict,
        only_r: Dict,
        only_p: Dict,
) -> HeteroData:
    """
    Build a PyG HeteroData reaction graph from SIMG node registry output.

    Parameters
    ----------
    mapped_rxn_smiles : full mapped reaction SMILES
    reg_r, reg_p      : full per-side registries from index_reaction()
    shared            : shared nodes dict  {key: {"feat_r","feat_p","delta"}}
    only_r            : nodes only in reactant  {key: tensor}
    only_p            : nodes only in product   {key: tensor}

    Returns
    -------
    SIMG extended CGR - HeteroData.
    map_id tensors per node type for cross-referencing.
    """
    data = HeteroData()

    # ── 1. Collect all node keys ─────────────────────────────────────────────
    all_keys = set(only_r) | set(only_p) | set(shared)

    atom_keys = sorted(k for k in all_keys if k[0] == "Atom")
    lp_keys   = sorted(k for k in all_keys if k[0] == "LP")
    bo_keys   = sorted(k for k in all_keys if k[0] == "BO")

    # Build index maps: key → integer node index within its type
    atom_idx = {k: i for i, k in enumerate(atom_keys)}
    lp_idx   = {k: i for i, k in enumerate(lp_keys)}
    bo_idx   = {k: i for i, k in enumerate(bo_keys)}

    # ── 2. Build node feature matrices ──────────────────────────────────────
    if atom_keys:
        data["atom"].x = torch.stack([
            _make_node_feat(k, shared, only_r, only_p) for k in atom_keys
        ])                                                    # (N_atom, 11)
        data["atom"].map_id = torch.tensor([k[1] for k in atom_keys])
    else:
        data["atom"].x = torch.zeros((0, ATOM_FEAT_DIM))
        data["atom"].map_id = torch.zeros(0, dtype=torch.long)

    if lp_keys:
        data["lp"].x = torch.stack([
            _make_node_feat(k, shared, only_r, only_p) for k in lp_keys
        ])                                                    # (N_lp, 13)
        # store (map_id, slot) for reference
        data["lp"].map_id = torch.tensor([k[1] for k in lp_keys])
        data["lp"].slot   = torch.tensor([k[2] for k in lp_keys])
    else:
        data["lp"].x = torch.zeros((0, LP_FEAT_DIM))
        data["lp"].map_id = torch.zeros(0, dtype=torch.long)
        data["lp"].slot   = torch.zeros(0, dtype=torch.long)

    if bo_keys:
        data["bo"].x = torch.stack([
            _make_node_feat(k, shared, only_r, only_p) for k in bo_keys
        ])                                                    # (N_bo, 17)
        data["bo"].mi   = torch.tensor([k[1] for k in bo_keys])
        data["bo"].mj   = torch.tensor([k[2] for k in bo_keys])
        data["bo"].slot = torch.tensor([k[3] for k in bo_keys])
    else:
        data["bo"].x = torch.zeros((0, BO_FEAT_DIM))
        data["bo"].mi   = torch.zeros(0, dtype=torch.long)
        data["bo"].mj   = torch.zeros(0, dtype=torch.long)
        data["bo"].slot = torch.zeros(0, dtype=torch.long)

    # ── 3. Atom-Atom bonds (CGR union) ───────────────────────────────────────
    rct_mol, pdt_mol = _split_rxn(mapped_rxn_smiles)
    cgr_bonds = _cgr_bonds(rct_mol, pdt_mol)

    aa_src, aa_dst, aa_feat = [], [], []
    for mi, mj, btype, flag in cgr_bonds:
        ki = ("Atom", mi)
        kj = ("Atom", mj)
        if ki not in atom_idx or kj not in atom_idx:
            continue   # atom not in registry (e.g. dropped H)
        i, j = atom_idx[ki], atom_idx[kj]
        feat = torch.cat([btype, flag])                       # (7,)
        # bidirectional
        aa_src += [i, j];  aa_dst += [j, i]
        aa_feat += [feat, feat]

    if aa_src:
        data["atom", "bond", "atom"].edge_index = torch.tensor(
            [aa_src, aa_dst], dtype=torch.long)
        data["atom", "bond", "atom"].edge_attr  = torch.stack(aa_feat)
    else:
        data["atom", "bond", "atom"].edge_index = torch.zeros((2,0), dtype=torch.long)
        data["atom", "bond", "atom"].edge_attr  = torch.zeros((0, BOND_EDGE_DIM))

    # ── 4. Atom-LP ownership edges ───────────────────────────────────────────
    owns_src, owns_dst = [], []
    for k in lp_keys:
        map_id = k[1]
        atom_k = ("Atom", map_id)
        if atom_k not in atom_idx:
            continue
        a_i  = atom_idx[atom_k]
        lp_i = lp_idx[k]
        owns_src.append(a_i);  owns_dst.append(lp_i)

    if owns_src:
        ei = torch.tensor([owns_src, owns_dst], dtype=torch.long)
        data["atom", "owns_lp",  "lp" ].edge_index = ei
        data["lp",   "owned_by", "atom"].edge_index = ei.flip(0)
    else:
        data["atom", "owns_lp",  "lp" ].edge_index = torch.zeros((2,0), dtype=torch.long)
        data["lp",   "owned_by", "atom"].edge_index = torch.zeros((2,0), dtype=torch.long)

    # ── 5. A2B edges (Atom → BO) ─────────────────────────────────────────────
    # Collect A2B features from both sides, prefer shared > only_r > only_p
    def _a2b_feat(a2b_key, shared, only_r, only_p):
        if a2b_key in shared:
            # A2B is an edge so shared dict has feat_r/feat_p/delta —
            # use reactant side feature
            return shared[a2b_key]["feat_r"].float()
        if a2b_key in only_r:
            return only_r[a2b_key].float()
        if a2b_key in only_p:
            return only_p[a2b_key].float()
        return None

    a2b_src, a2b_dst, a2b_feat_list = [], [], []
    a2b_rev_src, a2b_rev_dst = [], []

    all_a2b_keys = sorted(k for k in all_keys if k[0] == "A2B")
    for k in all_a2b_keys:
        # k = ("A2B", atom_map_id, bo_mi, bo_mj, bo_slot)
        _, mid, bo_mi, bo_mj, bo_slot = k
        atom_k = ("Atom", mid)
        bo_k   = ("BO", bo_mi, bo_mj, bo_slot)
        if atom_k not in atom_idx or bo_k not in bo_idx:
            continue
        feat = _a2b_feat(k, shared, only_r, only_p)
        if feat is None:
            continue
        a_i  = atom_idx[atom_k]
        bo_i = bo_idx[bo_k]
        a2b_src.append(a_i);  a2b_dst.append(bo_i)
        a2b_rev_src.append(bo_i);  a2b_rev_dst.append(a_i)
        a2b_feat_list.append(feat)

    if a2b_src:
        data["atom", "a2b",     "bo"  ].edge_index = torch.tensor([a2b_src,     a2b_dst],     dtype=torch.long)
        data["atom", "a2b",     "bo"  ].edge_attr  = torch.stack(a2b_feat_list)
        data["bo",   "a2b_rev", "atom"].edge_index = torch.tensor([a2b_rev_src, a2b_rev_dst], dtype=torch.long)
        data["bo",   "a2b_rev", "atom"].edge_attr  = torch.stack(a2b_feat_list)
    else:
        data["atom", "a2b",     "bo"  ].edge_index = torch.zeros((2,0), dtype=torch.long)
        data["atom", "a2b",     "bo"  ].edge_attr  = torch.zeros((0, A2B_EDGE_DIM))
        data["bo",   "a2b_rev", "atom"].edge_index = torch.zeros((2,0), dtype=torch.long)
        data["bo",   "a2b_rev", "atom"].edge_attr  = torch.zeros((0, A2B_EDGE_DIM))

    # ── 6. INT edges (LP ↔ BO, LP ↔ LP, BO ↔ BO) ───────────────────────────
    def _int_feat(int_key, shared, only_r, only_p):
        if int_key in shared:
            return shared[int_key]["feat_r"].float()
        if int_key in only_r:
            return only_r[int_key].float()
        if int_key in only_p:
            return only_p[int_key].float()
        return None

    # Separate into subtypes for proper HeteroData edge type naming
    int_edges = defaultdict(lambda: ([], [], []))   # (src_list, dst_list, feat_list)

    all_int_keys = sorted(k for k in all_keys if k[0] == "INT")
    for k in all_int_keys:
        _, key_A, key_B = k
        feat = _int_feat(k, shared, only_r, only_p)
        if feat is None:
            continue

        # determine which index maps to use
        def _get_idx(node_key):
            if node_key[0] == "LP":
                return "lp", lp_idx.get(node_key)
            if node_key[0] == "BO":
                return "bo", bo_idx.get(node_key)
            return None, None

        type_A, idx_A = _get_idx(key_A)
        type_B, idx_B = _get_idx(key_B)
        if idx_A is None or idx_B is None:
            continue

        edge_type = (type_A, "int", type_B)
        edge_type_rev = (type_B, "int", type_A)
        src_list, dst_list, feat_list = int_edges[edge_type]
        src_list.append(idx_A);  dst_list.append(idx_B)
        feat_list.append(feat)
        # bidirectional
        src_rev, dst_rev, feat_rev = int_edges[edge_type_rev]
        src_rev.append(idx_B);  dst_rev.append(idx_A)
        feat_rev.append(feat)

    for (et, (src_list, dst_list, feat_list)) in int_edges.items():
        data[et].edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        data[et].edge_attr  = torch.stack(feat_list)

    # fill missing INT edge types with empty tensors
    for et in [("lp","int","bo"), ("bo","int","lp"),
               ("lp","int","lp"), ("bo","int","bo")]:
        if et not in data.edge_types:
            data[et].edge_index = torch.zeros((2,0), dtype=torch.long)
            data[et].edge_attr  = torch.zeros((0, INT_EDGE_DIM))

    # ── 7. Store reaction SMILES for reference ───────────────────────────────
    data.rxn_smiles = mapped_rxn_smiles

    return data


# ════════════════════════════════════════════════════════════════════════════
# Quick inspection helper
# ════════════════════════════════════════════════════════════════════════════

def describe_cgr(data: HeteroData):
    """Print a human-readable summary of a CGR HeteroData object."""
    print("═" * 55)
    print("CGR HeteroData summary")
    print("═" * 55)
    print(f"Reaction: {data.rxn_smiles}")
    print()
    print("Nodes:")
    for ntype in data.node_types:
        n = data[ntype].x.shape[0]
        d = data[ntype].x.shape[1]
        print(f"  {ntype:6s}  {n:3d} nodes  feat_dim={d}")
    print()
    print("Edges:")
    for et in data.edge_types:
        e = data[et].edge_index.shape[1]
        has_feat = hasattr(data[et], "edge_attr") and data[et].edge_attr.shape[0] > 0
        fdim = data[et].edge_attr.shape[1] if has_feat else 0
        print(f"  {str(et):40s}  {e:4d} edges  feat_dim={fdim}")
    print("═" * 55)