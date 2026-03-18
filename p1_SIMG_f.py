import sys
# TODO change path
sys.path.append("/Users/qiuyue/Desktop/test1/simg")

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from simg.models import GNN_LP, GNN
from huggingface_hub import hf_hub_download

LP_CKPT_PATH = hf_hub_download("gomesgroup/simg", "lp_pred_model.ckpt")
INTERACTION_CHECKPOINT_PATH = hf_hub_download("gomesgroup/simg", "nbo_pred_model.ckpt")

lp_model  = GNN_LP.load_from_checkpoint(LP_CKPT_PATH, map_location="cpu", strict=False)
gnn_model = GNN.load_from_checkpoint(INTERACTION_CHECKPOINT_PATH, map_location="cpu", strict=False)

# ── Fix GATConv weight mismatch ───────────────────────────────────────────────
# The checkpoint was trained with an older PyG where GATConv had a single shared
# weight matrix called `lin`. Current PyG splits this into `lin_src` and `lin_dst`.
# Because strict=False silently skips the mismatch, `lin.weight` in each GATConv
# layer is left as a random PyTorch default initialisation — different every session.
# Fix: load the checkpoint manually and copy lin_src → lin for each GATConv layer.
def _fix_gatconv_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    own_state = model.state_dict()
    fixed = 0
    for key, param in state.items():
        if "lin_src.weight" in key:
            # e.g. "model.layers.0.lin_src.weight" → "model.layers.0.lin.weight"
            new_key = key.replace("lin_src.weight", "lin.weight")
            if new_key in own_state:
                own_state[new_key].copy_(param)
                fixed += 1
    model.load_state_dict(own_state, strict=False)
    print(f"GATConv weight fix: copied lin_src → lin for {fixed} layers in {model.__class__.__name__}")

_fix_gatconv_weights(gnn_model, INTERACTION_CHECKPOINT_PATH)

# ── eval mode ───────────────────────────────────────────────────────
# BatchNorm1d layers in the GNN heads compute live batch statistics in train mode
# (batch = all nodes of one molecule) → different stats every call.
# eval mode uses frozen running stats from training → fully deterministic.
lp_model.eval()
gnn_model.eval()

from simg.model_utils import get_initial_graph, predict_lps, get_atoms_per_lp
from simg.model_utils import get_final_graph, prepare_graph, make_preds_no_gt

ATOMS = ["H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi"]


# ── geometry helpers ──────────────────────────────────────────────────────────

def mol_to_symbols_coords_connectivity(mol, seed: int = 0):
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coordinates = np.array(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=float,
    )
    connectivity = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bond_type = 4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble())
        connectivity.append((i, j, bond_type))
    return symbols, coordinates, connectivity


def lp_prediction(model, symbols, coordinates, connectivity):
    mol_graph = get_initial_graph(symbols, coordinates, connectivity)
    n_lps, n_conj_lps = predict_lps(model, mol_graph)
    return n_lps, n_conj_lps, mol_graph


def build_data_for_nbo_forward(mol_graph, connectivity, n_lps_per_atom, n_conj_lps):
    graph = get_final_graph(mol_graph, connectivity, n_lps_per_atom, n_conj_lps)
    return prepare_graph(graph)


def nbo_prediction(model, data, threshold, use_threshold):
    if hasattr(data, "a2b_index"):
        a2b = data.a2b_index
        if not torch.is_tensor(a2b):
            a2b = torch.as_tensor(a2b, dtype=torch.long)
        if a2b.ndim == 2 and a2b.shape[0] != 2 and a2b.shape[1] == 2:
            a2b = a2b.T
        data.a2b_index = a2b.long()
    (preds_1, symbol_1, index_1), (a2b_preds, node_preds, int_preds) = \
        make_preds_no_gt(model, data, threshold=threshold, use_threshold=use_threshold)
    return (preds_1, symbol_1, index_1), (a2b_preds, node_preds, int_preds)


def normalize_edge_index(edge_index):
    """Ensure edge_index is a LongTensor of shape (2, E)."""
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.ndim == 1 and edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if edge_index.ndim == 2 and edge_index.shape[0] == 2:
        return edge_index.long()
    raise ValueError(f"Bad edge_index shape: {tuple(edge_index.shape)}")


# ── main entry point ──────────────────────────────────────────────────────────

def simg_run_one_molecule(
        mol: Chem.Mol,
        lp_model,
        gnn_model,
        threshold: float = 0.0,
        use_threshold: bool = False,
        seed: int = 0,
):
    """
    Run SIMG on one RDKit molecule (must already have Hs added via AddHs).

    Returns
    -------
    dict:
      data          : PyG Data (graph structure + is_atom / is_lp / is_bond masks)
      mol_map_ids   : list[int]  — RXNMapper atom-map numbers per atom (H → 0)
      atoms_per_lp  : list[int]  — atoms_per_lp[k] = RDKit atom index owning LP_k
      node_preds    : (N_total, 16) float tensor — GNN-predicted NBO properties
                        col  0- 3 : atom  targets (Charge, Core, Valence, Total)
                        col  4- 8 : LP    targets (s, p, d, f, occupancy)
                        col  9-15 : BO    targets (occupancy, s, p, d, f,
                                                   pol_diff, pol_coeff_diff)
      a2b_preds     : (E_a2b, 6) float tensor — per-atom contribution to each BO
                        (s, p, d, f, polarization, pol_coeff) — normalised to [0,1]
      int_preds     : (E_int, 3) float tensor — orbital interaction features
                        (perturbation_energy/100, energy_diff, Fock_element)
      orb_symbol    : list[str]  — orbital type labels for each orbital node
      orb_index     : list[int]  — global node indices for orbital nodes
      orb_link_matrix: (N_orb, N_orb) ndarray — predicted interaction matrix
    """
    mol_map_ids  = [a.GetAtomMapNum() for a in mol.GetAtoms()]   # H atoms → 0

    symbols, coordinates, connectivity = mol_to_symbols_coords_connectivity(mol, seed=seed)
    n_lps, n_conj_lps, mol_graph = lp_prediction(lp_model, symbols, coordinates, connectivity)
    data = build_data_for_nbo_forward(mol_graph, connectivity, n_lps, n_conj_lps)

    data.interaction_edge_index = normalize_edge_index(
        getattr(data, "interaction_edge_index", None)
    )

    n_atoms      = int(data.is_atom.sum())
    n_orb        = int(data.x.shape[0]) - n_atoms
    atoms_per_lp = get_atoms_per_lp(n_lps)   # positional LP → atom ownership

    if n_orb == 0:
        emb = gnn_model.get_embedding(data.x, data.edge_index, data.edge_attr)
        return {
            "data": data,
            "mol_map_ids": mol_map_ids,
            "atoms_per_lp": atoms_per_lp,
            "node_preds": emb.detach().cpu(),
            "a2b_preds": None,
            "int_preds": None,
            "orb_symbol": [],
            "orb_index": [],
            "orb_link_matrix": None,
        }

    (orb_link_matrix, orb_symbol, orb_index), (a2b_preds, node_preds, int_preds) = \
        nbo_prediction(gnn_model, data, threshold=threshold, use_threshold=use_threshold)

    return {
        "data": data,
        "mol_map_ids": mol_map_ids,
        "atoms_per_lp": atoms_per_lp,
        "node_preds": torch.as_tensor(node_preds),
        "a2b_preds": torch.as_tensor(a2b_preds) if a2b_preds is not None else None,
        "int_preds": torch.as_tensor(int_preds) if int_preds is not None else None,
        "orb_symbol": orb_symbol,
        "orb_index": orb_index,
        "orb_link_matrix": orb_link_matrix,
    }