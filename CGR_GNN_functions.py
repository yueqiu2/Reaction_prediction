from typing import Dict, Tuple, List, Any, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool
import torch.nn as nn

from rxnmapper import RXNMapper
from chython import smiles


# ------------
# CGR creator & CGR node/bone features design
# ------------

# 0) SMILE strings -> mapped -> CGR
def cgr_creator(rxn_smiles: list):
    '''
    :param rxn_smiles: reaction list in SMILE
    :return: CGR graph list
    '''

    # Reaction SMILES -> Mapped
    mapper = RXNMapper()
    mapped_results = mapper.get_attention_guided_atom_maps(rxn_smiles)

    # Mapped -> CGR
    cgr_list = [0] * len(rxn_smiles)
    for i, res in enumerate(mapped_results):
        mapped = res["mapped_rxn"]
        rxn_obj = smiles(mapped)
        cgr = rxn_obj.compose()
        cgr_list[i] = cgr
        # rxn = rxn_smiles[i]
        # print(f"Reaction {i}: {rxn}")
        # print(f"  mapped: {mapped}")

    return cgr_list


# 1) CGR Node feature design
# 1-1) Node index and features
def node_index_features(cgr):
    '''
    :param cgr: one CGR graph
    :return: map_id: List[int] atom-map-ids in node order
             node_id: Dict[map_id -> index]
             x: tensor
             map_id_tensor: LongTensor (N,) same order as x
    '''
    map_ids = sorted(cgr._atoms.keys())
    node_id = {m: i for i, m in enumerate(map_ids)}  # map[123...] -> idx[012...]

    x = torch.zeros((len(map_ids), 1), dtype=torch.long)
    for mid, i in node_id.items():
        atom = cgr._atoms[mid]
        x[i, 0] = int(atom.atomic_number)

    map_id_tensor = torch.tensor(map_ids, dtype=torch.long)

    return map_ids, node_id, x, map_id_tensor


# 2) CGR edge index and features design
# 2-0) All bonds in CGR [tuples in set], unique pairs e.g. (1,2)(2,3)...
def iter_cgr_pairs(cgr):
    seen = set()
    # mol._bonds is adjacency dict: if mol._bonds[a][b] exist then (a,b) are connected
    for a_map, nbrs in cgr._bonds.items():
        for b_map in nbrs.keys():
            key = (min(a_map, b_map), max(a_map, b_map))
            # Avoid duplicate
            if key in seen:
                continue
            seen.add(key)
            yield key


# 2-1) Bond order of ALL molecules on reactants/products side
def bond_order_in_side_dict(side_mols):
    '''
    :param side_mols: a tuple/list of molecule containers from reaction.reactants/products side
    :return: d: dict {(a,b): float...}, pair[key]: bond order[value]
    '''
    d = {}
    for mol in side_mols:
        # mol._bonds is adjacency dict: if mol._bonds[a][b] exist then (a,b) are connected
        for a_map, nbrs in mol._bonds.items():
            for b_map, bond in nbrs.items():
                key = (min(a_map, b_map), max(a_map, b_map))
                # bond.order should be a number here (molecule, not CGR)
                if bond.order is None:
                    continue
                d[key] = float(bond.order)
    return d


# 2-3) Bond features: before, after -> (dif) -> changed? broken/formed?
def bond_features_from_dict(a_map, b_map, react_dict, prod_dict):
    '''
    :param a_map/b_map: atom
    :param react_dict/prod_dict: dicts, bond orders of all bonds from one side.
    :return: Bond features of 1 bond, using dict lookups
    '''
    # Look up bond orders before/aftre from dicts
    key = (min(a_map, b_map), max(a_map, b_map))
    before = react_dict.get(key, 0.0)
    after = prod_dict.get(key, 0.0)

    # Change in bond order
    delta = after - before
    changed = 1.0 if before != after else 0.0  # if a bond changed
    broken = 1.0 if (before > 0 and after == 0) else 0.0  # if a bond broke
    formed = 1.0 if (before == 0 and after > 0) else 0.0  # if a bond formed

    return [float(before), float(after), float(delta), changed, broken, formed]


# 2-4) Build edge index and attributes for one reaction.
def edge_index_and_attr(cgr, node_id, reactants, products):
    '''
    :param cgr: CGRContainer (provides which atom pairs are connected in union graph)
    :param node_id: dict {atom_map -> node_index}
    :param reactants: rxn_obj.reactants (tuple of molecule containers)
    :param products:  rxn_obj.products  (tuple of molecule containers)
    :return edge_index: LongTensor [2, E]
            edge_attr:  FloatTensor [E, 6]  # subject to change
    Notes: both directions for each undirected bond (i->j and j->i).
    '''

    # Recompute bond orders on each side
    react_dict = bond_order_in_side_dict(reactants)
    prod_dict = bond_order_in_side_dict(products)

    edge_pairs = []
    edge_feats = []

    # Iterate undirected all CGR bonds
    for a_map, b_map in iter_cgr_pairs(cgr):
        i, j = node_id[a_map], node_id[b_map]

        feat = bond_features_from_dict(a_map, b_map, react_dict, prod_dict)

        # Store both directions for message passing
        edge_pairs.append([i, j]);
        edge_feats.append(feat)
        edge_pairs.append([j, i]);
        edge_feats.append(feat)

    if len(edge_pairs) == 0:
        raise ValueError("No edges built. Check CGR bonds / node_id mapping.")

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    return edge_index, edge_attr


# ------------
# Features augmentation 1, LP inference
# ------------
'''
LP inferences are obtained from SIMG GNN_LP network
SIMG_function.py constructs GNN inputs, [reaction SMILE] -> [input]
Also functions to run GNN
Output of GNN: for 1 reaction with N atoms on 1 side, (N, 10)
'''

# 0) For test: Concatenate LP inferences to CGR node feature [1+10]
def augment_cgr_node_features_with_lp(cgr, lp_by_map: Dict[int, torch.Tensor]) -> \
        Tuple[List[int], Dict[int, int], torch.Tensor]:
    '''
    :param lp_by_map_id: dictionary of LP inferences from GNN_LP
    :return: map_ids, node_id,
             Augmented x: Node feature matrix (N, 1 + 10)[atomic_number, LP features]
    '''

    # Base CGR node feature: atomic number
    map_ids, node_id, x_atomic, map_id_tensor = node_index_features(cgr)
    N = x_atomic.size(0)

    lp_dim = int(next(iter(lp_by_map.values())).numel()) if lp_by_map else 10
    lp_x = torch.zeros((N, lp_dim), dtype=torch.float)

    for i, map_id in enumerate(map_ids):
        v = lp_by_map.get(map_id)
        if v is not None:
            lp_x[i] = v.float()

    x = torch.cat([x_atomic.float(), lp_x], dim=1)
    return map_ids, node_id, x


# 1) Concatenate CGR + LP(react + prod + dif)
def augment_cgr_node_features_with_lp_both(
    cgr,
    lp_react: Dict[int, torch.Tensor],
    lp_prod: Dict[int, torch.Tensor],
    use_delta: bool = True
) -> Tuple[List[int], Dict[int, int], torch.Tensor]:
    '''
    :param lp_react/lp_prod: LP inferences dict of react/prod
           use_delta: [bool]. delta = lp_prod - lp_react
    :return: x with: [atomic_number | lp_react | lp_prod | (lp_prod - lp_react optional)]
             (N, 1 + 20), if use_delta: (N, 1 + 30)
    '''

    # CGR node feature: atomic number
    map_ids, node_id, x_atomic, map_id_tensor = node_index_features(cgr)
    N = x_atomic.size(0)

    # infer dim
    sample = None
    if lp_react:
        sample = next(iter(lp_react.values()))
    elif lp_prod:
        sample = next(iter(lp_prod.values()))
    lp_dim = int(sample.numel()) if sample is not None else 10

    xr = torch.zeros((N, lp_dim), dtype=torch.float)
    xp = torch.zeros((N, lp_dim), dtype=torch.float)

    # LP inferences
    for i, mid in enumerate(map_ids):
        vr = lp_react.get(mid)
        vp = lp_prod.get(mid)
        if vr is not None:
            xr[i] = vr.float()
        if vp is not None:
            xp[i] = vp.float()

    pieces = [x_atomic.float(), xr, xp]
    if use_delta:
        pieces.append(xp - xr)  # difference

    x = torch.cat(pieces, dim=1)
    return map_ids, node_id, x
