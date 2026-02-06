import sys

from rxnmapper import RXNMapper

# TODO require this: https://github.com/gomesgroup/simg,
# TODO download this to local directory and change path name here
sys.path.append("/Users/qiuyue/Desktop/test1/simg")

import torch
from torch_geometric.data import Data

import copy

import numpy as np
from chython import smiles
from rdkit import Chem
from rdkit.Chem import AllChem

# SIMG model
from simg.models import GNN_LP, GNN
from huggingface_hub import hf_hub_download

LP_CKPT_PATH = hf_hub_download("gomesgroup/simg", "lp_pred_model.ckpt")
INTERACTION_CHECKPOINT_PATH = hf_hub_download("gomesgroup/simg", "nbo_pred_model.ckpt")
# Load model
lp_model = GNN_LP.load_from_checkpoint(LP_CKPT_PATH, map_location="cpu", strict=False)
gnn_model = GNN.load_from_checkpoint(INTERACTION_CHECKPOINT_PATH, map_location="cpu", strict=False)


# Helper functions
from SIMG_f import mol_to_symbols_coords_connectivity, lp_prediction, \
    build_data_for_nbo_forward, nbo_prediction, \
    bondnode_to_atom_pair, simg_smiles_to_atom_mapdict, simg_smiles_to_bond_mapdict

# Atom vocabolary & lp model
ATOMS = ["H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi"]

# ============ SINGLE FUNCTION test ============= #

# mol_smile = 'CCBr.O'
mapped_mol_smiles = "[CH3:1][CH2:2][Br:3]"
# mapped_mol_smiles = "[OH2:4]"
mol = Chem.MolFromSmiles(mapped_mol_smiles)
mol = Chem.AddHs(mol)
# map id
# mol_map_ids = [a.GetAtomMapNum() for a in mol.GetAtoms()]
# symbols = [a.GetSymbol() for a in mol.GetAtoms()]
# print(mol_map_ids)
# print(symbols)


symbols, coordinates, connectivity = mol_to_symbols_coords_connectivity(mol)
print(connectivity)
# n_lps, n_conj_lps, mol_graph = lp_prediction(lp_model, symbols, coordinates, connectivity)
# #
# data = build_data_for_nbo_forward(mol_graph, connectivity, n_lps, n_conj_lps)
# #
# (preds_1, symbol_1, index_1), (a2b_preds, node_preds, int_preds) = \
#     nbo_prediction(gnn_model, data, threshold=0, use_threshold=True)
#
# print(symbol_1)
# print(index_1)
#print(node_preds.shape)


# # Bond -> dict test
# bondnode2atoms = bondnode_to_atom_pair(data.a2b_index)
#
# bond_feat_by_pair = simg_smiles_to_bond_mapdict(
#     node_preds=node_preds,              # (N_total, D)
#     bondnode2atoms=bondnode2atoms,       # {bondnode_idx: (atom_i, atom_j)}
#     mol_map_ids=mol_map_ids,             # atom map ids (including H=0)
#     index_1=index_1,
#     symbol_1=symbol_1,
#     include_h_bonds=False,               # for CGR attach
# )
#
# print(bond_feat_by_pair)
# print(node_preds[11])
# print(node_preds[12])



# ============ check ============= #

# structural check
def check_nbo_structural(data, preds_1, symbol_1, index_1, a2b_preds, node_preds, int_preds):
    n_atoms = int(data.is_atom.sum().item() if torch.is_tensor(data.is_atom) else int(np.sum(data.is_atom)))
    n_total = int(data.x.shape[0])
    n_orb = n_total - n_atoms

    assert preds_1.shape == (n_orb, n_orb), (preds_1.shape, n_orb)
    assert len(symbol_1) == n_orb, (len(symbol_1), n_orb)
    assert len(index_1) == n_orb, (len(index_1), n_orb)

    # node_preds usually aligns with all nodes
    assert node_preds.shape[0] in (n_total, n_orb), (node_preds.shape[0], n_total, n_orb)

    # If thresholding is enabled inside make_preds_no_gt, int_preds aligns with filtered interaction edges
    if hasattr(data, "interaction_edge_index") and data.interaction_edge_index is not None:
        e_int = int(data.interaction_edge_index.shape[1])
        assert int_preds.shape[0] == e_int, (int_preds.shape[0], e_int)

    print("✅ structural checks passed:",
          {"n_atoms": n_atoms, "n_total": n_total, "n_orb": n_orb,
           "n_int_edges": int(getattr(data, "interaction_edge_index").shape[1])})


# numeric check
def check_nbo_numeric(preds_1, a2b_preds, node_preds, int_preds):
    def stats(name, arr):
        arr = np.asarray(arr)
        print(f"{name:10s} shape={arr.shape} min={arr.min():.4g} max={arr.max():.4g} mean={arr.mean():.4g}")
        print(f"  nan={np.isnan(arr).any()} inf={np.isinf(arr).any()} zero_frac={(arr == 0).mean():.3f}")

    stats("preds_1", preds_1)
    stats("node", node_preds)
    stats("a2b", a2b_preds)
    stats("int", int_preds)


# interaction
def check_interaction_density(preds_1):
    n = preds_1.shape[0]
    n_possible = n * (n - 1)  # diagonal is zeroed
    n_selected = int(preds_1.sum())
    print("selected / possible:", n_selected, "/", n_possible, f"= {n_selected / n_possible:.3f}")

    # heuristic bands (not strict): you want “some but not all”
    if n_selected == 0:
        print("⚠️ No interactions selected: threshold too high or model not working.")
    if n_selected > 0.9 * n_possible:
        print("⚠️ Almost all selected: threshold too low (threshold=0 will do this).")


# result reproducible?
def check_reproducibility(data):
    data2 = copy.deepcopy(data)
    (predsA, _, _), (_, nodeA, intA) = nbo_prediction(gnn_model, data2, threshold=0.5, use_threshold=True)
    data3 = copy.deepcopy(data)
    (predsB, _, _), (_, nodeB, intB) = nbo_prediction(gnn_model, data3, threshold=0.5, use_threshold=True)

    print("preds equal:", np.array_equal(predsA, predsB))
    print("node close:", np.allclose(nodeA, nodeB, atol=1e-6))
    print("int close:", np.allclose(intA, intB, atol=1e-6))  # tolerance cannot be too small?


# check_nbo_structural(data, preds_1, symbol_1, index_1, a2b_preds, node_preds, int_preds)
# check_nbo_numeric(preds_1, a2b_preds, node_preds, int_preds)
# check_interaction_density(preds_1)
# check_reproducibility(data)


# ============= FULL FUNCTION TEST ============ #
# mapped_mol_smile = "[CH3:1][CH2:2][Br:3]"#
# feat_by_map = simg_smiles_to_mapdict(mapped_mol_smile, lp_model, gnn_model)
# print(feat_by_map)





