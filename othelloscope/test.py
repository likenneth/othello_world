import torch
import numpy as np

import transformer_lens.utils as utils
from data import get_othello

from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
)

board_seqs_int = torch.tensor(np.load("board_seqs_int_small.npy"), dtype=torch.long)
board_seqs_string = torch.tensor(
    np.load("board_seqs_string_small.npy"), dtype=torch.long
)

num_games, length_of_game = board_seqs_int.shape
print(
    "Number of games:",
    num_games,
)
print("Length of game:", length_of_game)

print(str(board_seqs_int[0]))

cfg = HookedTransformerConfig(
    n_layers=8,
    d_model=512,
    d_head=64,
    n_heads=8,
    d_mlp=2048,
    d_vocab=61,
    n_ctx=59,
    act_fn="gelu",
    normalization_type="LNPre",
)
model = HookedTransformer(cfg).cuda()

sd = utils.download_file_from_hf(
    "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
)
# champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
model.load_state_dict(sd)

output = model(board_seqs_int[0, :59])

print(str(output[:, -1, 1:]))
