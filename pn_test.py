from p1_SIMG_f import simg_run_one_molecule, lp_model, gnn_model
from p2_node_indexing import index_reaction
from p3_cgr import build_cgr

reg_r, reg_p, shared, only_r, only_p = index_reaction(
    "[CH3:1][CH2:2][Br:3].[OH2:4]>>[CH3:1][CH2:2][OH:4].[BrH:3]",
    simg_fn=simg_run_one_molecule,
    lp_model=lp_model, gnn_model=gnn_model,
    drop_h=True,
)

data = build_cgr(
    mapped_rxn_smiles="[CH3:1][CH2:2][Br:3].[OH2:4]>>[CH3:1][CH2:2][OH:4].[BrH:3]",
    reg_r=reg_r, reg_p=reg_p,
    shared=shared, only_r=only_r, only_p=only_p,
)
print(data)