# %%
from neel.imports import *

# %%
import os
import math
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
from data.othello import OthelloBoardState

torch.set_grad_enabled(False)
# %%
make_dataset = False

if make_dataset:
    from mingpt.dataset import CharDataset
    from data import get_othello
    print("Making dataset")
    othello = get_othello(ood_num=-1, data_root=None, wthor=True)
    train_dataset = CharDataset(othello)

    full_seqs = list(filter(lambda x: len(x)==60, train_dataset.data.sequences))
    print(len(full_seqs))
    board_seqs = torch.tensor(full_seqs)
    print(board_seqs.numel())
    
    # n = 50000
    # board_seqs = torch.zeros((n, 60), dtype=int)
    # for c, seq in enumerate(tqdm(othello.sequences)):
    #     board_seqs[c, :len(seq)] = torch.tensor(seq)
    #     if c == n-1:
    #         break
    
    board_seqs_string = board_seqs
    print(board_seqs_string.numel())
    
    board_seqs_int = board_seqs_string.clone()
    board_seqs_int[board_seqs_string < 29] += 1
    board_seqs_int[(board_seqs_string >= 29) & (board_seqs_string <= 34)] -= 1
    board_seqs_int[(board_seqs_string > 34)] -= 3
    rand = torch.randint(0, 1000000, (20,))
    print(board_seqs_int.flatten()[rand])
    print(board_seqs_string.flatten()[rand])
    # torch.save(board_seqs, "board_seqs.pt")
    
    indices = torch.randperm(len(board_seqs_int))
    board_seqs_int = board_seqs_int[indices]
    board_seqs_string = board_seqs_string[indices]
    torch.save(board_seqs_int, "board_seqs_int.pth")
    torch.save(board_seqs_string, "board_seqs_string.pth")
else:
    board_seqs_int = torch.load("board_seqs_int.pth")
    board_seqs_string = torch.load("board_seqs_string.pth")
imshow(board_seqs_int[:5], title="Board Seqs Int Test")
imshow(board_seqs_string[:5], title="Board Seqs String Test")
# %%
itos = {
    0: -100,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    25: 24,
    26: 25,
    27: 26,
    28: 29,
    29: 30,
    30: 31,
    31: 32,
    32: 33,
    33: 34,
    34: 37,
    35: 38,
    36: 39,
    37: 40,
    38: 41,
    39: 42,
    40: 43,
    41: 44,
    42: 45,
    43: 46,
    44: 47,
    45: 48,
    46: 49,
    47: 50,
    48: 51,
    49: 52,
    50: 53,
    51: 54,
    52: 55,
    53: 56,
    54: 57,
    55: 58,
    56: 59,
    57: 60,
    58: 61,
    59: 62,
    60: 63,
}

stoi = {
    -100: 0,
    -1: 0,
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21,
    21: 22,
    22: 23,
    23: 24,
    24: 25,
    25: 26,
    26: 27,
    29: 28,
    30: 29,
    31: 30,
    32: 31,
    33: 32,
    34: 33,
    37: 34,
    38: 35,
    39: 36,
    40: 37,
    41: 38,
    42: 39,
    43: 40,
    44: 41,
    45: 42,
    46: 43,
    47: 44,
    48: 45,
    49: 46,
    50: 47,
    51: 48,
    52: 49,
    53: 50,
    54: 51,
    55: 52,
    56: 53,
    57: 54,
    58: 55,
    59: 56,
    60: 57,
    61: 58,
    62: 59,
    63: 60,
}
# %%
stoi_indices = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    29,
    30,
    31,
    32,
    33,
    34,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
]
alpha = "ABCDEFGH"


def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"


board_labels = list(map(to_board_label, stoi_indices))
# %%
def str_to_int(s):
    return stoi[s] - 1


# %%
# train_dataset.vocab_size, train_dataset.block_size == (61, 59)
# mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)
# min_model = GPT(mconf)
# min_model.load_state_dict(torch.load("gpt_synthetic.ckpt"))
# %%
import transformer_lens.utils as utils

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
model = HookedTransformer(cfg)


sd = utils.download_file_from_hf(
    "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
)
# champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
model.load_state_dict(sd)
# %%
# board_seqs_int = torch.load("board_seqs_int.pth")
# board_seqs_string = torch.load("board_seqs_string.pth")
# %%
board = OthelloBoardState()
board.update(board_seqs_string[0, :5].tolist())
analyse_object(board)
print(board.get_valid_moves())
# %%
def get_valid_moves(sequence):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.tolist()
    board = OthelloBoardState()
    return board.get_gt(sequence, "get_valid_moves")


get_valid_moves(board_seqs_string[0])
# %%
def make_plot_state(board):
    state = np.copy(board.state).flatten()
    valid_moves = board.get_valid_moves()
    next_move = board.get_next_hand_color()
    # print(next_move, valid_moves)
    for move in valid_moves:
        state[move] = next_move - 0.5
    return state


def add_counter(fig, position, color):
    is_black = color > 0
    row = position // 8
    col = position % 8
    fig.layout.shapes += (
        dict(
            type="circle",
            x0=col - 0.2,
            y0=row - 0.2,
            x1=col + 0.2,
            y1=row + 0.2,
            fillcolor="black" if is_black else "white",
            line_color="green",
            line_width=0.5,
        ),
    )
    return fig


def counter_shape(position, color, mode="normal"):
    is_black = color > 0
    row = position // 8
    col = position % 8
    shape = dict(
        type="circle",
        fillcolor="black" if is_black else "white",
    )
    if mode == "normal":
        shape.update(
            x0=col - 0.2,
            y0=row - 0.2,
            x1=col + 0.2,
            y1=row + 0.2,
            line_color="green",
            line_width=0.5,
        )
    elif mode == "flipped":
        shape.update(
            x0=col - 0.22,
            y0=row - 0.22,
            x1=col + 0.22,
            y1=row + 0.22,
            line_color="purple",
            line_width=3,
        )
    elif mode == "new":
        shape.update(
            line_color="red",
            line_width=4,
            x0=col - 0.25,
            y0=row - 0.25,
            x1=col + 0.25,
            y1=row + 0.25,
        )
    return shape


def plot_board(moves, return_fig=False):
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()
    board = OthelloBoardState()
    states = []
    states.append(make_plot_state(board))
    for move in moves:
        board.umpire(move)
        states.append(make_plot_state(board))
    states = np.stack(states, axis=0)
    fig = imshow(
        states.reshape(-1, 8, 8),
        color_continuous_scale="Geyser",
        aspect="equal",
        return_fig=True,
        animation_frame=0,
        y=["a", "b", "c", "d", "e", "f", "g", "h"],
        x=["0", "1", "2", "3", "4", "5", "6", "7"],
        animation_index=[
            f"{i+1} ({'W' if i%2==0 else 'B'}) [{to_board_label(moves[i]) if i>=0 else 'X'} -> {to_board_label(moves[i+1]) if i<len(moves)-1 else 'X'}]"
            for i in range(-1, len(moves))
        ],
        animation_name="Move",
    )
    fig.update_traces(
        text=[[str(i + 8 * j) for i in range(8)] for j in range(8)],
        texttemplate="%{text}",
    )
    for c, frame in enumerate(fig.frames):
        for i in range(64):
            if states[c].flatten()[i] == 1:
                frame = add_counter(frame, i, True)
            elif states[c].flatten()[i] == -1:
                frame = add_counter(frame, i, False)
    fig.layout.shapes = fig.frames[0].layout.shapes
    if return_fig:
        return fig
    else:
        fig.show()


# plot_board(board_seqs_string[0, :5])
# %%
def add_ring(fig, position, color):
    is_black = color > 0
    row = position // 8
    col = position % 8
    offset = 0.3
    fig.layout.shapes += (
        dict(
            type="rect",
            x0=col - offset,
            y0=row - offset,
            x1=col + offset,
            y1=row + offset,
            line_color="black" if is_black else "red",
            line_width=5,
            fillcolor=None,
        ),
    )
    return fig


def plot_board_log_probs(moves, logits, return_fig=False, use_counters=False):
    logits = logits.squeeze(0)
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()
    # print(moves)
    assert len(moves) == len(logits)
    board = OthelloBoardState()
    states = []
    # states.append(make_plot_state(board))
    for move in moves:
        board.umpire(move)
        states.append(make_plot_state(board))
    states = np.stack(states, axis=0)

    log_probs = logits.log_softmax(dim=-1)
    log_probs_template = torch.zeros((len(moves), 64)).cuda() - 100
    if log_probs.shape[-1] == 61:
        log_probs_template[:, stoi_indices] = log_probs[:, 1:]
    else:
        log_probs_template[:, stoi_indices] = log_probs[:, :]
    log_probs_template = log_probs_template.reshape(-1, 8, 8)

    fig = imshow(
        log_probs_template,
        color_continuous_scale="Blues",
        zmin=-6.0,
        zmax=0.0,
        aspect="equal",
        return_fig=True,
        animation_frame=0,
        y=["a", "b", "c", "d", "e", "f", "g", "h"],
        x=["0", "1", "2", "3", "4", "5", "6", "7"],
        animation_index=[
            f"{i+1} ({'W' if i%2==0 else 'B'}) [{to_board_label(moves[i])} -> {to_board_label(moves[i+1]) if i<len(moves)-1 else 'X'}]"
            for i in range(len(moves))
        ],
        animation_name="Move",
    )
    # fig.update_traces(text=[[str(i+8*j) for i in range(8)] for j in range(8)], texttemplate="%{text}")
    for c, frame in enumerate(tqdm(fig.frames)):
        text = []
        shapes = []
        for i in range(64):
            text.append("")
            counter_text = "O" if moves[c] != i else "X"
            if states[c].flatten()[i] == 1:
                if use_counters:
                    shapes.append(counter_shape(i, True))
                else:
                    # black = red
                    text[
                        -1
                    ] = f"<b style='font-size: 24em; color: red; '>{counter_text}</b>"
            elif states[c].flatten()[i] == -1:
                if use_counters:
                    shapes.append(counter_shape(i, False))
                else:
                    # white = green
                    text[
                        -1
                    ] = f"<b style='font-size: 24em; color: green;'>{counter_text}</b>"
            else:
                if states[c].flatten()[i] > 0.2:
                    text[
                        -1
                    ] = f"<span style='font-size: 12em; '>{to_board_label(i)}</span>"
                    # print(i, c, "b")
                    # frame = add_ring(frame, i, True)
                elif states[c].flatten()[i] < -0.2:
                    text[
                        -1
                    ] = f"<span style='font-size: 12em; color: white'>{to_board_label(i)}</span>"
                    # print(i, c, "w")
                    # frame = add_ring(frame, i, False)
        frame.layout.shapes = tuple(shapes)
        frame.data[0]["text"] = np.array(text).reshape(8, 8)
        frame.data[0]["texttemplate"] = "%{text}"
        frame.data[0][
            "hovertemplate"
        ] = "<b>%{y}%{x}</b><br>log prob: %{z}<br>prob=%{customdata}<extra></extra>"
        frame.data[0]["customdata"] = to_numpy(log_probs_template[c].exp())
    # print(states)
    fig.layout.shapes = fig.frames[0].layout.shapes
    fig.data[0]["text"] = fig.frames[0].data[0]["text"]
    fig.data[0]["texttemplate"] = fig.frames[0].data[0]["texttemplate"]
    fig.data[0]["customdata"] = fig.frames[0].data[0]["customdata"]
    fig.data[0]["hovertemplate"] = fig.frames[0].data[0]["hovertemplate"]
    if return_fig:
        return fig
    else:
        fig.show()


logits = model(board_seqs_int[0, :10].unsqueeze(0))
# fig = plot_board_log_probs(board_seqs_string[0, :10], logits, return_fig=True, use_counters=True)
# fig.show()
# fig = plot_board_log_probs(
#     board_seqs_string[0, :10], logits, return_fig=True, use_counters=False
# )
# fig.show()
# logits.shape
# print(fig.layout)
# %%
def move_to_player(move):
    return "W" if move % 2 == 0 else "B"


# %%
logits, cache = model.run_with_cache(board_seqs_int[0, :-1], remove_batch_dim=True)
big_fig = plot_board_log_probs(
    board_seqs_string[0, :-1], logits[...], return_fig=True, use_counters=False
)
# big_fig.show()
# %%
resid_decomp, labels = cache.decompose_resid(-1, return_labels=True)
resid_decomp = cache.apply_ln_to_stack(resid_decomp, -1)
decomp_logits = resid_decomp @ model.W_U[:, 1:]
print(decomp_logits.shape)
for move in [4, 10, 20, 40]:
    temp_fig = imshow(
        torch.zeros(8, 8),
        color_continuous_scale="Blues",
        zmin=-6.0,
        zmax=0.0,
        aspect="equal",
        return_fig=True,
        y=["a", "b", "c", "d", "e", "f", "g", "h"],
        x=["0", "1", "2", "3", "4", "5", "6", "7"],
    )
    temp_fig.add_trace(big_fig.frames[move - 1].data[0])
    temp_fig.data = temp_fig.data[-1:]
    temp_fig.layout = big_fig.frames[move - 1].layout
    temp_fig.update_layout(
        title=f"Move {move}: {to_board_label(board_seqs_string[0, move-1].item())} ({move_to_player(move-1)}) ->  {to_board_label(board_seqs_string[0, move].item())} ({move_to_player(move)})",
        title_x=0.5,
        title_xanchor="center",
    )
    temp_fig.show()

    imshow(
        decomp_logits[:, move - 1],
        x=board_labels,
        y=labels,
        title=f"Logit lens for board state after {move}",
        zmax=3,
        zmin=-3.0,
    )
# %%
str_tokens = list(map(to_board_label, board_seqs_int[0, :-1].tolist()))
# psvelte.
# %%
if False:
    print("Pre-Filter", len(board_seqs_string))
    filtered_indices = (board_seqs_string == -100).sum(-1) == 0
    board_seqs_string = board_seqs_string[filtered_indices]
    board_seqs_int = board_seqs_int[filtered_indices]
    print("Post-Filter", len(board_seqs_string))
    indices = torch.randperm(len(board_seqs_int))
    board_seqs_int = board_seqs_int[indices]
    board_seqs_string = board_seqs_string[indices]

# %%
num_seqs = 50
big_logits, big_cache = model.run_with_cache(board_seqs_int[:num_seqs, :-1])
small_logits, small_cache = model.run_with_cache(board_seqs_int[:5, :-1])

big_log_probs = big_logits.log_softmax(-1)
big_logits = big_logits[..., 1:]
big_log_probs = big_log_probs[:, 1:]
small_log_probs = small_logits.log_softmax(-1)
small_logits = small_logits[..., 1:]
small_log_probs = small_log_probs[:, 1:]
# %%
valid_moves_list = [get_valid_moves(board_seqs_string[i]) for i in range(num_seqs)]
# %%
is_valid_move = torch.zeros_like(big_logits, dtype=torch.bool, device="cpu")
for i in range(num_seqs):
    for j in range(model.cfg.n_ctx):
        valid_moves = valid_moves_list[i][j]
        valid_moves = [str_to_int(m) for m in valid_moves]
        is_valid_move[i, j, valid_moves] = True
print(is_valid_move[5, 19], valid_moves_list[5][19])
is_valid_move = is_valid_move.cuda()

# %%
def tensor_to_board(tensor):
    board = torch.zeros(
        size=tensor.shape[:-1] + (64,), device=tensor.device, dtype=tensor.dtype
    )
    if tensor.shape[-1] == 61:
        tensor = tensor[..., :-1]
    board[..., stoi_indices] = tensor
    return board.reshape(board.shape[:-1] + (8, 8))


imshow(tensor_to_board(is_valid_move[5, 19]), aspect="equal", return_fig=True).show()
print([to_board_label(i) for i in valid_moves_list[5][19]])
# %%
print("Max", is_valid_move.sum(-1).max())
print("Min", is_valid_move.sum(-1).min())
# %%
stacked_big_resid, labels = big_cache.decompose_resid(-1, return_labels=True)
stacked_big_resid = big_cache.apply_ln_to_stack(stacked_big_resid, -1)
big_decomp_logits = stacked_big_resid @ model.W_U[:, 1:]
print(big_decomp_logits.shape)
# %%
def t(n):
    return torch.tensor(n, device="cuda", dtype=torch.float32)


def sanity_check_tensor(tensor):
    print(tensor.shape, tensor.min(), tensor.max())


correct_logits = (
    torch.where(is_valid_move[None, :, :, :], big_decomp_logits, t(100)).min(-1).values
)
incorrect_logits = (
    torch.where(~is_valid_move[None, :, :, :], big_decomp_logits, t(-100))
    .max(-1)
    .values
)
sanity_check_tensor(correct_logits)
sanity_check_tensor(incorrect_logits)

# %%
scale = 3.0
imshow(
    (correct_logits - incorrect_logits).mean(-2).cpu(),
    title="Min Correct - Max Incorrect",
    y=labels,
    xaxis="Position",
    zmax=3.0,
    zmin=-3.0,
)
# %%

num_valid_moves = is_valid_move.sum(-1)
correct_logits_ave = (
    torch.where(is_valid_move[None, :, :, :], big_decomp_logits, t(0)).sum(-1)
    / num_valid_moves
)
incorrect_logits_ave = torch.where(
    ~is_valid_move[None, :, :, :], big_decomp_logits, t(-0)
).sum(-1) / (60 - num_valid_moves)
sanity_check_tensor(correct_logits_ave)
sanity_check_tensor(incorrect_logits_ave)
scale = 3.0
imshow(
    (correct_logits_ave - incorrect_logits_ave).mean(-2).cpu(),
    title="Mean Correct - Mean Incorrect",
    y=labels,
    xaxis="Position",
    zmax=3.0,
    zmin=-3.0,
)

# %%

big_patterns = big_cache.stack_activation("pattern")
# %%

text_index = 0
plot_board(board_seqs_string[text_index, :-1])
# %%
fig = plot_board_log_probs(
    board_seqs_string[0, :-1], small_logits[0, :], use_counters=False, return_fig=True
)
fig.write_html("Game0.html", include_plotlyjs="cdn")
fig.show()
# %%

patterns = big_patterns[:, text_index, :]
print(patterns.shape)
patterns = einops.rearrange(patterns, "layer head dest src -> dest src (layer head)")
str_tokens = list(map(to_board_label, board_seqs_string[text_index, :-1].tolist()))

head_patterns = pysvelte.AttentionMulti(
    attention=patterns[:, :, -24:],
    tokens=[f"{tok}_{i} " for i, tok in enumerate(str_tokens)],
    head_labels=[
        f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ][-24:],
)
head_patterns.show()
# %%
stacked_small_resid, labels = small_cache.get_full_resid_decomposition(
    -1, return_labels=True, expand_neurons=False
)
stacked_small_resid = small_cache.apply_ln_to_stack(stacked_small_resid, -1)
small_decomp_logits = stacked_small_resid @ model.W_U[:, 1:]
print(small_decomp_logits.shape)

num_valid_moves = is_valid_move.sum(-1)
correct_logits_ave = (
    torch.where(
        is_valid_move[None, : len(small_logits), :, :], small_decomp_logits, t(0)
    ).sum(-1)
    / num_valid_moves[: len(small_logits)]
)
incorrect_logits_ave = (
    torch.where(
        ~is_valid_move[None, : len(small_logits), :, :], small_decomp_logits, t(-0)
    ).sum(-1)
    / (60 - num_valid_moves)[: len(small_logits)]
)
sanity_check_tensor(correct_logits_ave)
sanity_check_tensor(incorrect_logits_ave)
scale = 3.0
imshow(
    (correct_logits_ave - incorrect_logits_ave).mean(-2).cpu(),
    title="Mean Correct - Mean Incorrect",
    y=labels,
    xaxis="Position",
    zmax=3.0,
    zmin=-3.0,
)
# %%


def to_int(x):
    # print("\t", x)
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_int(x.item())
    elif (
        isinstance(x, list) or isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)
    ):
        return [to_int(i) for i in x]
    elif isinstance(x, int):
        return stoi[x]
    elif isinstance(x, str):
        x = x.upper()
        return to_int(to_string(x))


def to_string(x):
    """Confusingly, maps it to an int, but a board pos label not a token label (token labels have 0 == pass, and middle board cells don't exist)"""
    # print("\t", x)
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_string(x.item())
    elif (
        isinstance(x, list) or isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)
    ):
        return [to_string(i) for i in x]
    elif isinstance(x, int):
        return itos[x]
    elif isinstance(x, str):
        x = x.upper()
        return 8 * alpha.index(x[0]) + int(x[1])


def to_label(x, from_int=True):
    # print("\t", x)
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return to_label(x.item(), from_int=from_int)
    elif (
        isinstance(x, list) or isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)
    ):
        return [to_label(i, from_int=from_int) for i in x]
    elif isinstance(x, int):
        if from_int:
            return to_board_label(to_string(x))
        else:
            return to_board_label(x)
    elif isinstance(x, str):
        return x


int_to_label = to_label
string_to_label = partial(to_label, from_int=False)
str_to_label = string_to_label
print(to_int([0, 63, 30]))
print(to_int("A4"))
print(to_string(torch.tensor([1, 60, 30])))
print(to_string(["A4", "D3", "D7"]))
print(to_label(torch.tensor([1, 60, 30])))
print(to_label(torch.tensor([1, 60, 30]), from_int=False))
# print(to_int(["A4", "D3", "D7"]))

# %%
def moves_to_state(moves):
    # moves is a list of string entries (ints)
    state = np.zeros((8, 8), dtype=bool)
    for move in moves:
        state[move // 8, move % 8] = 1.0
    return state


# print(moves_to_state([26, 18]))
int_labels = (
    list(range(1, 28))
    + ["X", "X"]
    + list(range(28, 34))
    + ["X", "X"]
    + list(range(34, 61))
)


def plot_single_board(moves, return_fig=False, title=None):
    # moves is a list of string entries (ints)
    if isinstance(moves, torch.Tensor):
        moves = moves.tolist()
    if isinstance(moves[0], str):
        moves = to_string(moves)
    board = OthelloBoardState()
    if len(moves) > 1:
        board.update(moves[:-1])

    prev_state = np.copy(board.state)
    prev_player = board.next_hand_color
    prev_valid_moves = board.get_valid_moves()
    board.umpire(moves[-1])
    next_state = np.copy(board.state)
    next_player = board.next_hand_color
    next_valid_moves = board.get_valid_moves()

    empty = (prev_state == 0) & (next_state == 0)
    new = (prev_state == 0) & (next_state != 0)
    flipped = (prev_state != 0) & (next_state != prev_state) & (~new)
    prev_valid = moves_to_state(prev_valid_moves)
    next_valid = moves_to_state(next_valid_moves)

    state = np.copy(next_state)
    state[flipped] *= 0.9
    state[prev_valid] = 0.25 * prev_player
    state[next_valid] = 0.5 * next_player
    state[new] = 0.9 * prev_player

    logits = model(torch.tensor(to_int(moves)).cuda().unsqueeze(0)).cpu()
    log_probs = logits.log_softmax(-1)
    lps = torch.zeros(64) - 15.0
    lps[stoi_indices] = log_probs[0, -1, 1:]

    if title is None:
        title = f"Board State After {'Black' if prev_player==1 else 'White'} Plays {to_label(moves[-1], from_int=False)}"

    fig = imshow(
        state,
        color_continuous_scale="Geyser",
        title=title,
        y=[i for i in alpha],
        x=[str(i) for i in range(8)],
        aspect="equal",
        return_fig=True,
    )
    fig.data[0]["hovertemplate"] = "<b>%{y}%{x}</b><br>%{customdata}<extra></extra>"

    shapes = []
    texts = []
    for i in range(64):
        texts.append("")
        if empty.flatten()[i]:
            texts[-1] = to_label(i, from_int=False)
        elif flipped.flatten()[i]:
            shapes.append(counter_shape(i, prev_player == 1, mode="flipped"))
        elif new.flatten()[i]:
            shapes.append(counter_shape(i, prev_player == 1, mode="new"))
        elif prev_state.flatten()[i] != 0:
            shapes.append(counter_shape(i, prev_state.flatten()[i] == 1, mode="normal"))
        else:
            raise ValueError(i)
    fig.layout.shapes = tuple(shapes)
    fig.data[0]["text"] = np.array(texts).reshape(8, 8)
    fig.data[0]["texttemplate"] = "%{text}"
    fig.data[0]["customdata"] = np.array(
        [f"LP:{lps[i].item():.4f}<br>I:{int_labels[i]}<br>S:{i}" for i in range(64)]
    ).reshape(8, 8)

    if return_fig:
        return fig
    else:
        fig.show()
    return


plot_single_board(["D2", "C4"])
# %%
cutoff = 20
moves = board_seqs_string[5, :cutoff]
moves = str_to_label(moves)
print(moves)
plot_single_board(moves, title="Initial")
# plot_single_board(moves+["G0"])

clean_labels = moves + ["H0"]
plot_single_board(clean_labels, title="Clean Board after H0")
corr_labels = moves + ["G0"]
plot_single_board(corr_labels, title="Corrupted Board after G0")
clean_tokens = to_int(clean_labels)
corr_tokens = to_int(corr_labels)
logit_index = to_int("C0") - 1
print(logit_index)
clean_logits, clean_cache = model.run_with_cache(
    torch.tensor(clean_tokens).cuda().unsqueeze(0)
)
clean_logits = clean_logits[0, -1, 1:]
corr_logits, corr_cache = model.run_with_cache(
    torch.tensor(corr_tokens).cuda().unsqueeze(0)
)
corr_logits = corr_logits[0, -1, 1:]


def logit_metric(logits):
    if len(logits.shape) == 3:
        logits = logits[0]
    if len(logits.shape) == 2:
        logits = logits[-1]
    if len(logits) == 61:
        logits = logits[1:]
    return logits[logit_index]


clean_baseline = logit_metric(clean_logits).detach()
corr_baseline = logit_metric(corr_logits).detach()
print("Clean Baseline:", clean_baseline.item())
print("corr Baseline:", corr_baseline.item())
metric = lambda logits: (logit_metric(logits) - corr_baseline) / (
    clean_baseline - corr_baseline
)

# %%
from transformer_lens import patching

every_block_act_patch_result = patching.get_act_patch_block_every(
    model, torch.tensor(corr_tokens).unsqueeze(0).cuda(), clean_cache, metric
)

imshow(
    every_block_act_patch_result,
    facet_col=0,
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
    title="Activation Patching Per Block",
    xaxis="Position",
    yaxis="Layer",
    zmax=1,
    zmin=-1,
    x=[f"{tok}_{i}" for i, tok in enumerate(clean_labels)],
)
# %%
head_labels = [
    f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
]
imshow(
    (
        clean_cache.stack_activation("pattern")[:, 0, :, -1]
        - corr_cache.stack_activation("pattern")[:, 0, :, -1]
    ).reshape(64, -1),
    y=head_labels,
    x=clean_labels,
)
# %%
logit_vec = model.W_U[:, 1 + logit_index]
line_list = []
for mlp_index in range(8):
    mlp_diff = (
        clean_cache["post", mlp_index, "mlp"][0, -1]
        - corr_cache["post", mlp_index, "mlp"][0, -1]
    )
    line_list.append(mlp_diff * (model.blocks[mlp_index].mlp.W_out @ logit_vec))
line(line_list, title="Direct Logit Attribution of Diffed Neurons")
# %%

line(
    [big_cache["post", i, "mlp"][:, 5:-5].mean([0, 1]).sort().values for i in range(8)],
    title="Means of MLP Neurons (Center Pos)",
)
line(
    [big_cache["post", i, "mlp"][:, :].mean([0, 1]).sort().values for i in range(8)],
    title="Means of MLP Neurons (All Pos)",
)
# %%
for mlp_index in [4, 5, 6, 7]:
    cutoff = 0.2
    neuron_means = big_cache["post", mlp_index, "mlp"][:, :].mean([0, 1])
    neuron_indices = neuron_means > cutoff
    W_out = model.blocks[mlp_index].mlp.W_out[neuron_indices]
    print(f"{neuron_indices.sum().item()} Neurons above cutoff {cutoff}")
    imshow(
        W_out @ model.W_U[:, 1:],
        y=list(map(str, torch.arange(2048)[neuron_indices.cpu()].tolist())),
        yaxis="Neuron",
        title=f"Direct Logit Attr of top neurons in Layer {mlp_index}",
        xaxis="Move",
        x=int_to_label(list(range(1, 61))),
    )
# %%
neuron_index = 1339
# %%
def head_hook(z, hook, head_index, layer):
    z[0, -1, head_index, :] = clean_cache["z", layer][0, -1, head_index, :]
    return z


head_patches = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device="cuda")
for layer in tqdm(range(model.cfg.n_layers)):
    for head_index in range(model.cfg.n_heads):
        logits = model.run_with_hooks(
            torch.tensor(corr_tokens).cuda().unsqueeze(0),
            fwd_hooks=[
                (
                    utils.get_act_name("z", layer),
                    partial(head_hook, head_index=head_index, layer=layer),
                )
            ],
        )
        head_patches[layer, head_index] = metric(logits)
imshow(
    head_patches, title="Head Patching", xaxis="Head", yaxis="Layer", zmax=1, zmin=-1
)

# %%
def head_abl_hook(z, hook, head_index, layer):
    z[0, -1, head_index, :] = corr_cache["z", layer][0, -1, head_index, :]
    return z


head_abl_patches = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device="cuda")
for layer in tqdm(range(model.cfg.n_layers)):
    for head_index in range(model.cfg.n_heads):
        logits = model.run_with_hooks(
            torch.tensor(clean_tokens).cuda().unsqueeze(0),
            fwd_hooks=[
                (
                    utils.get_act_name("z", layer),
                    partial(head_abl_hook, head_index=head_index, layer=layer),
                )
            ],
        )
        head_abl_patches[layer, head_index] = metric(logits)
imshow(
    head_abl_patches,
    title="Resample Ablation",
    xaxis="Head",
    yaxis="Layer",
    zmax=1,
    zmin=-1,
)

# %%
def head_hook(z, hook, head_index, layer):
    z[0, -1, head_index, :] = clean_cache["z", layer][0, -1, head_index, :]
    return z


head_patches = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device="cuda")
for layer in tqdm(range(model.cfg.n_layers)):
    for head_index in range(model.cfg.n_heads):
        logits = model.run_with_hooks(
            torch.tensor(corr_tokens).cuda().unsqueeze(0),
            fwd_hooks=[
                (
                    utils.get_act_name("z", layer),
                    partial(head_hook, head_index=head_index, layer=layer),
                )
            ],
        )
        head_patches[layer, head_index] = metric(logits)
imshow(
    head_patches, title="Head Patching", xaxis="Head", yaxis="Layer", zmax=1, zmin=-1
)
# %%
neuron_diffs = (
    clean_cache.stack_activation("post")[:, 0, -1]
    - corr_cache.stack_activation("post")[:, 0, -1]
)
line(neuron_diffs, title="Neuron Act Diffs")
# %%
cutoff = 200
sorted_neuron_vals, sorted_neuron_indices = neuron_diffs.abs().sort(
    dim=-1, descending=True
)
fig = line(sorted_neuron_vals, return_fig=True)
fig.add_vline(x=cutoff, line_width=1, line_dash="dash", line_color="black")
fig.show()
top_neurons = sorted_neuron_indices[:, :cutoff]

# %%
def neuron_hook(post, hook, neuron, layer):
    post[0, -1, neuron] = corr_cache["post", layer][0, -1, neuron]
    return post

neuron_abl_patches = torch.zeros((model.cfg.n_layers, cutoff), device="cuda")
for layer in tqdm(range(model.cfg.n_layers)):
    for ni in range(cutoff):
        logits = model.run_with_hooks(
            torch.tensor(clean_tokens).cuda().unsqueeze(0),
            fwd_hooks=[
                (
                    utils.get_act_name("post", layer),
                    partial(neuron_hook, neuron=top_neurons[layer, ni].item(), layer=layer),
                )
            ],
        )
        neuron_abl_patches[layer, ni] = metric(logits)
imshow(
    neuron_abl_patches - 1, title="Top Neuron Resample Ablation", xaxis="Top Neuron Index", yaxis="Layer", zmax=1, zmin=-1, hover=top_neurons
)

# %%
def neuron_hook(post, hook, neuron, layer):
    post[0, -1, neuron] = clean_cache["post", layer][0, -1, neuron]
    return post

neuron_patches = torch.zeros((model.cfg.n_layers, cutoff), device="cuda")
for layer in tqdm(range(model.cfg.n_layers)):
    for ni in range(cutoff):
        logits = model.run_with_hooks(
            torch.tensor(corr_tokens).cuda().unsqueeze(0),
            fwd_hooks=[
                (
                    utils.get_act_name("post", layer),
                    partial(neuron_hook, neuron=top_neurons[layer, ni].item(), layer=layer),
                )
            ],
        )
        neuron_patches[layer, ni] = metric(logits)
imshow(
    neuron_patches, title="Top Neuron Tracing", xaxis="Top Neuron Index", yaxis="Layer", zmax=1, zmin=-1, hover=top_neurons
)

# %%
# OK, let's zoom in on that neuron(?!)
ni = 1393
layer = 5
def patch_neuron(post, hook):
    post[0, -1, ni] = clean_cache["post", layer][0, -1, ni]
model.blocks[layer].mlp.hook_post.add_hook(patch_neuron)
patched_logits, patched_cache = model.run_with_cache(torch.tensor(corr_tokens).cuda().unsqueeze(0))
print("Patched Metric", metric(patched_logits))
print("Patched Logit", logit_metric(patched_logits))
print("Patched Log Prob", patched_logits.log_softmax(dim=-1)[0, -1, logit_index+1])
# %%
direct_logit_attr = []
for cache in [clean_cache, corr_cache, patched_cache]:
    resid_stack, labels = cache.decompose_resid(-1, return_labels=True)
    resid_stack = cache.apply_ln_to_stack(resid_stack) 
    direct_logit_attr.append(resid_stack[:, 0, -1, :] @ model.W_U[:, logit_index+1])
line(direct_logit_attr, line_labels=["clean", "corr", "patch"], title="Direct Logit Attribution by Layer", x=np.array(labels))
# %%
direct_logit_attr = []
for cache in [clean_cache, corr_cache, patched_cache]:
    direct_logit_attr.append(cache["post", 7][0, -1] * (model.blocks[7].mlp.W_out @ model.W_U[:, logit_index + 1]) / 1.5)
    
line(direct_logit_attr, line_labels=["clean", "corr", "patch"], title="Direct Logit Attribution by Neuron")
# %%
unembed_labels = ["Pass"] + to_label(list(range(1, 61)))
line(model.blocks[layer].mlp.W_out[ni] @ model.blocks[6].mlp.W_in, title="Neuron 5 1393 vs layer 6 neurons")
line(model.blocks[layer].mlp.W_out[ni] @ model.blocks[6].mlp.W_in, title="Neuron 5 1393 vs layer 7 neurons")
line(model.blocks[layer].mlp.W_out[ni] @ model.W_U, title="Neuron 5 1393 vs Unembed", x=np.array(unembed_labels))
# %%
def seq_to_state_stack(str_moves):
    if isinstance(str_moves, torch.Tensor):
        str_moves = str_moves.tolist()
    board = OthelloBoardState()
    states = []
    for move in str_moves:
        board.umpire(move)
        states.append(np.copy(board.state))
    states = np.stack(states, axis=0)
    return states
state_stack = torch.tensor(np.stack([seq_to_state_stack(seq) for seq in board_seqs_string[:50, :-1]]))
print(state_stack.shape)
# %%
big_mlp_post = big_cache.stack_activation("post")
print(big_mlp_post.shape)
# %%
alternating = torch.tensor([1 if i%2 == 0 else -1 for i in range(59)])
state_stack_flipped = state_stack * alternating[None, :, None, None]
# %%
layer =5 
ni = 1393
acts = big_mlp_post[layer, :, :, ni]
indices = acts.flatten() > acts.flatten().quantile(.99)
states = state_stack.reshape(-1, 8, 8)[indices]
states_flipped = state_stack_flipped.reshape(-1, 8, 8)[indices]
print(states.shape)
imshow(states.abs().mean(0), title=f"Mean Abs State for Top Acts L{layer}N{ni}")
imshow(states.mean(0), title=f"Mean State for Top Acts L{layer}N{ni}")
imshow(states_flipped.abs().mean(0), title=f"Mean Abs State Flipped for Top Acts L{layer}N{ni}")
imshow(states_flipped.mean(0), title=f"Mean State Flipped for Top Acts L{layer}N{ni}")
# %%
neuron_values_sorted, neuron_indices_sorted = big_mlp_post.mean([1, 2]).sort(dim=-1, descending=True)
top_neuron_indices = neuron_indices_sorted[:, :50]
# %%
flipped_states = []
abs_states = []
for layer in range(8):
    flipped_states.append([])
    abs_states.append([])
    for i in range(10):
        ni = top_neuron_indices[layer, i]
        acts = big_mlp_post[layer, :, :, ni]
        indices = acts.flatten() > acts.flatten().quantile(.98)
        states_flipped = state_stack_flipped.reshape(-1, 8, 8)[indices]
        flipped_states[-1].append(states_flipped.mean(0))
        abs_states[-1].append(states_flipped.abs().mean(0))
flipped_states_stack = torch.stack([torch.stack(flip) for flip in flipped_states])
imshow(flipped_states_stack, zmin=-1, zmax=1, title="Mean Flipped State for Top Neurons", y=[i for i in alpha], x = [str(i) for i in range(8)], aspect="equal", facet_col=1, facet_name="Neuron", animation_frame=0, animation_name="Layer")
absed_states_stack = torch.stack([torch.stack(absed) for absed in abs_states])
imshow(absed_states_stack, zmin=-1, zmax=1, title="Mean absed State for Top Neurons", y=[i for i in alpha], x = [str(i) for i in range(8)], aspect="equal", facet_col=1, facet_name="Neuron", animation_frame=0, animation_name="Layer")
# %%
# Some post hoc work
big_game = board_seqs_int[:500]
big_logits, big_cache = model.run_with_cache(big_game[:, :-1])

big_state_stack = [seq_to_state_stack(board_seqs_string[i]) for i in range(500)]

big_state_stack = torch.tensor(np.stack(big_state_stack, axis=0))

print(big_state_stack.shape)




neuron_acts = big_cache["post", 5][:, :, 1393]
print(neuron_acts.shape)

print(big_state_stack.shape)
# %%
alternating = -torch.tensor([1 if i%2 == 0 else -1 for i in range(60)])

flipped_big_state_stack = alternating[None, :, None, None] * big_state_stack

imshow(flipped_big_state_stack[0, 0])

c0 = flipped_big_state_stack[:, :, 2, 0]
d1 = flipped_big_state_stack[:, :, 3, 1]
e2 = flipped_big_state_stack[:, :, 4, 2]

label = (c0==0) & (d1==-1) & (e2==1)

imshow(label[:30])
imshow(neuron_acts[:30])

histogram(neuron_acts.flatten(), color=label.flatten())
# %%
df = pd.DataFrame({"acts":neuron_acts.flatten().tolist(), "label":label[:, :-1].flatten().tolist()})
px.histogram(df, x="acts", color="label")
# %%
