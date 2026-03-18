# ═══════════════════════════════════════════════════════════════════════
# SIMG_test_crosssession.py
#
# Run this script TWICE in separate terminal sessions.
# First run:  generates the cache and prints feature values.
# Second run: loads the same cache and prints the same feature values.
# The numbers must be identical.
#
# Usage:
#   python SIMG_test_crosssession.py
# ═══════════════════════════════════════════════════════════════════════
import torch
from rdkit import Chem
from SIMG_f1 import simg_run_one_molecule, lp_model, gnn_model
from node_indexing1 import index_all_nodes, get_node_features
# from conformer_cache1 import generate_and_save_conformer, load_conformer

MOL_SMI   = "[CH3:1][CH2:2][Br:3]"
# CACHE_DIR = "./conformer_cache"   # permanent — same dir both sessions
SEP       = "─" * 55

# ── Step 1: generate cache if it doesn't exist yet, otherwise reuse ───
print(SEP)
# path = generate_and_save_conformer(MOL_SMI, CACHE_DIR, seed=0, overwrite=False)
print(f"Conformer cache: {path}")

# ── Step 2: always load from cache ────────────────────────────────────
# mol     = load_conformer(MOL_SMI, CACHE_DIR)
run_out = simg_run_one_molecule(mol, lp_model, gnn_model,
                                threshold=0.0, use_threshold=False, seed=0)
reg     = index_all_nodes(run_out, drop_h=True)

# ── Step 3: print values — must be identical across sessions ──────────
print(SEP)
atom_t = reg[("Atom", 1)]
lp_t   = reg[("LP",   3, 0)]
bo_t   = reg[("BO",   1, 2, 0)]

print(f"Atom [0:4]  = {atom_t[0:4].tolist()}")
print(f"LP   [4:9]  = {lp_t[4:9].tolist()}")
print(f"BO   [9:16] = {bo_t[9:16].tolist()}")
print(SEP)
print("Run this again in a new session — the numbers above must be identical.")