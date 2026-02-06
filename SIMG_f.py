'''
Pipeline:


'''

import sys
from rxnmapper import RXNMapper

# TODO change path
sys.path.append("/Users/qiuyue/Desktop/test1/simg")

import torch
from torch_geometric.data import Data
from collections import defaultdict

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
from simg.model_utils import get_initial_graph, predict_lps
from simg.model_utils import get_final_graph, prepare_graph, make_preds_no_gt

# Atom vocabolary & lp model
ATOMS = ["H", "B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi"]


# get symbols, coordinates, connectivity
def mol_to_symbols_coords_connectivity(mol, seed: int = 0):
    # IF TEST THIS FUNCTION ONLY, ADD 2 LINES BELOW
    '''
    mol = Chem.MolFromSmiles(mol)    # convert str -> RDkit object
    mol = Chem.AddHs(mol)  # H is considered in SIMG GNN
    '''

    # 3D
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    # Get symbols of atoms
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    # Get coordinates of atoms
    coordinates = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=float,
    )

    # Get connectivity: [a,b,type], bone type in one-hot
    connectivity = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        if b.GetIsAromatic():
            bond_type = 4
        else:
            bond_type = int(b.GetBondTypeAsDouble())  # 1/2/3
        connectivity.append((i, j, bond_type))

    return symbols, coordinates, connectivity


# LP prediction
def lp_prediction(model, symbols, coordinates, connectivity):
    # build initial graph
    mol_graph = get_initial_graph(symbols, coordinates, connectivity)  # return for next
    # LP GNN prediction
    n_lps, n_conj_lps = predict_lps(model, mol_graph)

    return n_lps, n_conj_lps, mol_graph


# GNN inputs
def build_data_for_nbo_forward(mol_graph, connectivity, n_lps_per_atom, n_conj_lps):
    # build graph (atoms + LP + BND)
    graph = get_final_graph(mol_graph, connectivity, n_lps_per_atom, n_conj_lps)
    # Tensorize -> PyG Data
    data = prepare_graph(graph)

    return data


# GNN prediction
def nbo_prediction(model, data, threshold, use_threshold):
    '''
    Interaction of orbital nodes (LP/BND) expressed in probability (sigmoid)
    threshold is to cutoff low prob orbital nodes
    if threshold=0, means every node interactions are considered -> might too many

    return:
        preds: interaction, binary if use threshold
        symbol, index: universal nodes symbol/type & index
        a2b_preds, node_preds, int_preds: bond-atom, node, interaction summary
    '''

    (preds_1, symbol_1, index_1), (a2b_preds, node_preds, int_preds) = \
        make_preds_no_gt(model, data, threshold=threshold, use_threshold=use_threshold)

    return (preds_1, symbol_1, index_1), (a2b_preds, node_preds, int_preds)


# ============ SIMG on mol - Atom -> map dict ============= #
'''
The mapped smile reactions eg, [CH3:1][CH2:2][Br:3].[OH2:4] >> [CH3:1][CH2:2][OH:4].[BrH:3]
Each atoms in molecules already has [map id] attached to it, with CGR has the same [map id] setting
[map id] will be the universal id linking CGR & SIMG
'''


# For case with no interaction
def normalize_edge_index(edge_index):
    """
    Ensure edge_index is a LongTensor of shape (2, E).
    Accepts None or weird empties like tensor([]).
    """
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)

    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)

    # If it's a flat empty tensor: tensor([])
    if edge_index.ndim == 1 and edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    # If it's already (2, E)
    if edge_index.ndim == 2 and edge_index.shape[0] == 2:
        return edge_index.long()

    # Anything else is unexpected
    raise ValueError(f"Bad edge_index shape: {tuple(edge_index.shape)}")


# SIMG for one molecule
def simg_mol_to_atom_mapdict(mol, lp_model, gnn_model, \
                             threshold=0, use_threshold=True, seed=0):
    '''
    :param mapped_mol_smiles: mapped reaction -> mapped molecules
    :return: feat_by_map, mol dict{int: tensor}, each atom -> atom_id: tensor
    '''

    # atom-map-ids in RDKit atom order (same order you build coords/connectivity from)
    mol_map_ids = [a.GetAtomMapNum() for a in mol.GetAtoms()]  # H usually 0

    # 2) build symbols/coords/connectivity
    symbols, coordinates, connectivity = mol_to_symbols_coords_connectivity(mol, seed=seed)

    # 3) LP → graph → Data
    n_lps, n_conj_lps, mol_graph = lp_prediction(lp_model, symbols, coordinates, connectivity)
    data = build_data_for_nbo_forward(mol_graph, connectivity, n_lps, n_conj_lps)

    # guard: handle empty interaction cases
    n_atoms = int(data.is_atom.sum())
    n_orb = int(data.x.shape[0] - n_atoms)
    # IF INTERACTION = 0 eg. HBr
    data.interaction_edge_index = normalize_edge_index(getattr(data, "interaction_edge_index", None))
    E_int = data.interaction_edge_index.shape[1]

    if n_orb == 0 or E_int == 0:
        emb = gnn_model.get_embedding(data.x, data.edge_index, data.edge_attr)
        atom_emb = emb[data.is_atom.bool()].detach().cpu().float()
        feat_by_map = {}
        for i, mid in enumerate(mol_map_ids):
            if mid > 0:
                feat_by_map[mid] = atom_emb[i]
        return feat_by_map

    # 4) NBO prediction
    # TODO take more later
    (_, _, _), (_, node_preds, _) = nbo_prediction(gnn_model, data, threshold=threshold, use_threshold=use_threshold)

    # 5) Slice atom rows and map them to map_id
    node_preds = torch.as_tensor(node_preds)
    atom_preds = node_preds[data.is_atom.bool()]  # (N_atoms, d)

    assert atom_preds.shape[0] == len(mol_map_ids), (
        atom_preds.shape[0], len(mol_map_ids)
    )

    feat_by_map = {}
    for i, mid in enumerate(mol_map_ids):
        if mid > 0:  # ignore unmapped H
            feat_by_map[mid] = atom_preds[i].detach().cpu().float()

    return feat_by_map, node_preds, mol_map_ids


# RDkit JUST ONCE
def simg_smiles_to_atom_mapdict(mapped_mol_smiles, lp_model, gnn_model, **kwargs):
    mol = Chem.MolFromSmiles(mapped_mol_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {mapped_mol_smiles}")
    mol = Chem.AddHs(mol)
    return simg_mol_to_atom_mapdict(mol, lp_model, gnn_model, **kwargs)


# ============ mol -> BOND -> map dict (with prediction result reused) ============= #

# extract bond from a2b_index, recorded in map-id
def bondnode_to_atom_pair(a2b_index):
    """
    a2b_index: from Data, tensor of shape (2*Nbonds, 2) with rows [atom_idx, bondnode_idx]
    returns: dict {bondnode_idx: (atom_i, atom_j)}
    """
    tmp = defaultdict(list)
    for atom_i, bond_k in a2b_index.tolist():
        tmp[bond_k].append(atom_i)

    out = {}
    for bond_k, atoms in tmp.items():
        assert len(atoms) == 2, (bond_k, atoms)
        out[bond_k] = (atoms[0], atoms[1])
    return out


# ensure consistency of universal node index in SIMG
# compare index_1 & symbol_1 with a2b_index
def assert_bnd_indices_consistent(bondnode2atoms, index_1, symbol_1):
    """
    index_1: list/np array of orbital node indices
    symbol_1: list of same length with 'LP'/'BND'
    """
    if index_1 is not None and symbol_1 is not None:
        bnd_from_symbols = {int(idx) for idx, sym in zip(index_1, symbol_1) if sym == "BND"}
        bnd_from_a2b = set(bondnode2atoms.keys())

        missing = bnd_from_a2b - bnd_from_symbols
        wrong_type = {int(idx) for idx, sym in zip(index_1, symbol_1) if sym != "BND"} & bnd_from_a2b

        if missing:
            raise AssertionError(f"Some bondnode indices from a2b_index are not labeled BND: {sorted(missing)}")
        if wrong_type:
            # This would mean a node used as "bondnode" is labeled LP, which indicates a bug upstream
            raise AssertionError(f"Some a2b bondnode indices are not BND type: {sorted(wrong_type)}")


# extract bond orbital features from SIMG node result
def simg_smiles_to_bond_mapdict(node_preds, bondnode2atoms, mol_map_ids, \
                                index_1=None, symbol_1=None, include_h_bonds=False, expect_unique=True):
    """
    node_preds: Tensor (N_total, D)  - can be full node_pred rows or embeddings
    bondnode2atoms: dict {bondnode_idx: (atom_i, atom_j)}
    mol_map_ids: list length N_atoms, map_id=0 for H
    index_1: iterable of orbital node indices (optional, for asserts)
    symbol_1: iterable of same length with labels 'LP'/'BND' (optional)
    include_h_bonds: if False, skip any bond where an endpoint has map_id=0
    expect_unique: if True, error on duplicate (map_i,map_j) keys

    returns: dict {(min_map, max_map): Tensor(D)}
    """

    # check consistency
    assert_bnd_indices_consistent(bondnode2atoms, index_1, symbol_1)

    node_preds = torch.as_tensor(node_preds)
    out = {}

    for bondnode_idx, (ai, aj) in bondnode2atoms.items():
        mi = int(mol_map_ids[ai])
        mj = int(mol_map_ids[aj])

        if not include_h_bonds and (mi == 0 or mj == 0):
            continue

        key = (mi, mj) if mi < mj else (mj, mi)
        vec = node_preds[bondnode_idx].detach().cpu().float()  # full row, no slicing

        if expect_unique and key in out:
            raise ValueError(f"Duplicate bond key {key} encountered; check mapping or merging logic.")
        out[key] = vec

    return out


# ============ reaction -> mol -> map dict ============= #

# split mapped reaction to molecules
def split_mapped_rxn(mapped_rxn_smiles: str):
    left, right = mapped_rxn_smiles.split(">>")
    react_mols = [s for s in left.split(".") if s]
    prod_mols = [s for s in right.split(".") if s]
    return react_mols, prod_mols


def run_simg_on_reaction(mapped_rxn_smiles, lp_model, gnn_model, threshold=0, use_threshold=True, seed=0):
    react_mols, prod_mols = split_mapped_rxn(mapped_rxn_smiles)

    feat_r = {}
    for smi in react_mols:
        d, _, _ = simg_smiles_to_atom_mapdict(smi, lp_model, gnn_model,
                                              threshold=threshold, use_threshold=use_threshold, seed=seed)
        feat_r.update(d)

    feat_p = {}
    for smi in prod_mols:
        d, _, _ = simg_smiles_to_atom_mapdict(smi, lp_model, gnn_model,
                                              threshold=threshold, use_threshold=use_threshold, seed=seed)
        feat_p.update(d)

    return feat_r, feat_p

# rxn = "[CH3:1][CH2:2][Br:3].[OH2:4]>>[CH3:1][CH2:2][OH:4].[BrH:3]"
# feat_r, feat_p = run_simg_on_reaction(rxn, lp_model, gnn_model)
# print(sorted(feat_r.keys()))
# print(sorted(feat_p.keys()))
