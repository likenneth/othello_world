import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig
from mech_interp_othello_utils import OthelloBoardState
import einops
import torch
from tqdm import tqdm
import numpy as np
from fancy_einsum import einsum

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
board_seqs_int = torch.tensor(np.load("board_seqs_int_small.npy")).long()
board_seqs_string = torch.tensor(np.load("board_seqs_string_small.npy"))
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


state_stack = torch.tensor(
    np.stack([seq_to_state_stack(seq) for seq in board_seqs_string[:50, :-1]])
)
print(state_stack.shape)
# %%

# %%
layer = 6
batch_size = 100
lr = 1e-4
wd = 0.01
pos_start = 5
pos_end = model.cfg.n_ctx - 5
length = pos_end - pos_start
options = 3
rows = 8
cols = 8
num_epochs = 2
num_games = 100000
x = 0
y = 2
probe_name = "main_linear_probe"
# The first mode is blank or not, the second mode is next or prev GIVEN that it is not blank
modes = 3
alternating = torch.tensor([1 if i%2 == 0 else -1 for i in range(length)], device="cuda")


def state_stack_to_one_hot(state_stack):
    one_hot = torch.zeros(
        modes, # blank vs color (mode)
        state_stack.shape[0], # num games
        state_stack.shape[1], # num moves
        rows, # rows
        cols, # cols
        options, # the two options
        device=state_stack.device,
        dtype=torch.int,
    )
    one_hot[:, ..., 0] = state_stack == 0
    one_hot[:, ..., 1] = state_stack == -1
    one_hot[:, ..., 2] = state_stack == 1
    return one_hot
state_stack_one_hot = state_stack_to_one_hot(state_stack)
print(state_stack_one_hot.shape)
print((state_stack_one_hot[:, 0, 17, 4:9, 2:5]))
print((state_stack[0, 17, 4:9, 2:5]))
# %%
linear_probe = torch.randn(
    modes, model.cfg.d_model, rows, cols, options, requires_grad=False, device="cuda"
)/np.sqrt(model.cfg.d_model)
linear_probe.requires_grad = True
optimiser = torch.optim.AdamW([linear_probe], lr=lr, betas=(0.9, 0.99), weight_decay=wd)


for epoch in range(num_epochs):
    full_train_indices = torch.randperm(num_games)
    for i in tqdm(range(0, num_games, batch_size)):
        indices = full_train_indices[i:i+batch_size]
        games_int = board_seqs_int[indices]
        games_str = board_seqs_string[indices]
        state_stack = torch.stack(
            [torch.tensor(seq_to_state_stack(games_str[i])) for i in range(batch_size)]
        )
        state_stack = state_stack[:, pos_start:pos_end, :, :]

        state_stack_one_hot = state_stack_to_one_hot(state_stack).cuda()
        with torch.inference_mode():
            _, cache = model.run_with_cache(games_int.cuda()[:, :-1], return_type=None)
            resid_post = cache["resid_post", layer][:, pos_start:pos_end]
        probe_out = einsum(
            "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
            resid_post,
            linear_probe,
        )
        # print(probe_out.shape)

        # acc_blank = (probe_out[0].argmax(-1) == state_stack_one_hot[0].argmax(-1)).float().mean()
        # acc_color = ((probe_out[1].argmax(-1) == state_stack_one_hot[1].argmax(-1)) * state_stack_one_hot[1].sum(-1)).float().sum()/(state_stack_one_hot[1]).float().sum()

        probe_log_probs = probe_out.log_softmax(-1)
        probe_correct_log_probs = einops.reduce(
            probe_log_probs * state_stack_one_hot,
            "modes batch pos rows cols options -> modes pos rows cols",
            "mean"
        ) * options # Multiply to correct for the mean over options
        loss_even = -probe_correct_log_probs[0, 0::2].mean(0).sum() # note that "even" means odd in the game framing, since we offset by 5 moves lol
        loss_odd = -probe_correct_log_probs[1, 1::2].mean(0).sum()
        loss_all = -probe_correct_log_probs[2, :].mean(0).sum()
        
        loss = loss_even + loss_odd + loss_all
        loss.backward() # it's important to do a single backward pass for mysterious PyTorch reasons, so we add up the losses - it's per mode and per square.

        optimiser.step()
        optimiser.zero_grad()
torch.save(linear_probe, f"{probe_name}.pth")
# %%
# %%