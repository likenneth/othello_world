# %%
from tl_othello_utils import *
# %%
# train_dataset.vocab_size, train_dataset.block_size == (61, 59)
# mconf = GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512)
# min_model = GPT(mconf)
# min_model.load_state_dict(torch.load("gpt_synthetic.ckpt"))
# %%
torch.set_grad_enabled(False)
# %%
plot_single_board(["D2", "C4"])
plot_board_log_probs(to_string(["D2", "C4"]), model(torch.tensor(to_int(["D2", "C4"]))))
plot_board(["D2", "C4"])

# %%
board_seqs_int = torch.load("board_seqs_int.pth")
board_seqs_string = torch.load("board_seqs_string.pth")
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


def state_stack_to_one_hot(state_stack):
    one_hot = torch.zeros(
        state_stack.shape[0],
        state_stack.shape[1],
        8,
        8,
        3,
        device=state_stack.device,
        dtype=torch.int,
    )
    one_hot[..., 0] = state_stack == 0
    one_hot[..., 1] = state_stack == -1
    one_hot[..., 2] = state_stack == 1
    return one_hot
state_stack_one_hot = state_stack_to_one_hot(state_stack)
print((state_stack_one_hot[0, 10, 4:6, 2:5]))
print((state_stack[0, 10, 4:6, 2:5]))
# %%
layer = 6
batch_size = 100
lr = 1e-4
wd = 0.01
# even, odd, all
modes = 3
pos_start = 5
pos_end = model.cfg.n_ctx - 5
length = pos_end - pos_start
options = 3
rows = 8
cols = 8
num_epochs = 5
num_games = 4000000
x = 0
y = 2

alternating = torch.tensor([1 if i%2 == 0 else -1 for i in range(length)], device="cuda")
# %%
linear_probe = torch.randn(
    modes, model.cfg.d_model, rows, cols, options, requires_grad=False, device="cuda"
)/np.sqrt(model.cfg.d_model)
linear_probe.requires_grad = True
optimiser = torch.optim.AdamW([linear_probe], lr=lr, betas=(0.9, 0.99), weight_decay=wd)
# %%
# wandb.init(project="othello", name="linear-probe")
# %%
# for epoch in range(num_epochs):
#     full_train_indices = torch.randperm(num_games)
#     for i in tqdm(range(0, num_games, batch_size)):
#         indices = full_train_indices[i:i+batch_size]
#         games_int = board_seqs_int[indices]
#         games_str = board_seqs_string[indices]
#         state_stack = torch.stack(
#             [torch.tensor(seq_to_state_stack(games_str[i])) for i in range(batch_size)]
#         )
#         state_stack = state_stack[:, pos_start:pos_end, :, :]



#         state_stack_one_hot = state_stack_to_one_hot(state_stack)

#         _, cache = model.run_with_cache(games_int.cuda()[:, :-1], return_type=None)
#         resid_post = cache["resid_post", layer][:, pos_start:pos_end]
#         probe_out = einsum(
#             "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
#             resid_post,
#             linear_probe,
#         )
#         # print(probe_out.shape)

#         probe_log_probs = probe_out.log_softmax(-1)
#         probe_correct_log_probs = einops.reduce(
#             probe_log_probs * state_stack_one_hot.cuda(),
#             "modes batch pos rows cols options -> modes pos rows cols",
#             "mean"
#         ) * 3 # * 3 to correct for the mean
#         loss_even = -probe_correct_log_probs[0, 0::2].mean(0).sum()
#         loss_odd = -probe_correct_log_probs[1, 1::2].mean(0).sum()
#         loss_all = -probe_correct_log_probs[2, :].mean(0).sum()
#         loss = loss_even + loss_odd + loss_all
#         loss.backward()

#         wandb.log({"loss_even":loss_even.item(), "loss_odd":loss_odd.item(), "loss_all":loss_all.item(),
#         "loss_even_D1":-probe_correct_log_probs[0, 0::2].mean(0)[3, 1].item(), "loss_odd_D1":-probe_correct_log_probs[1, 1::2].mean(0)[3, 1].item(), "loss_all_D1":-probe_correct_log_probs[2, :].mean(0)[3, 1].item(),})

#         optimiser.step()
#         optimiser.zero_grad()
#     torch.save(linear_probe, f"linear_probe_epoch{epoch}.pth")
# %%
# %%
linear_probe = torch.load("linear_probe_epoch1.pth")
print(linear_probe.shape)
# %%
indices = torch.arange(num_games+7894, num_games+7894 +100)
games_int = board_seqs_int[indices]
games_str = board_seqs_string[indices]
state_stack = torch.stack(
    [torch.tensor(seq_to_state_stack(games_str[i])) for i in range(100)]
)
state_stack = state_stack[:, :-1, :, :]

state_stack_one_hot = state_stack_to_one_hot(state_stack)

logits, cache = model.run_with_cache(games_int.cuda()[:, :-1], return_type="logits")
resid_post = cache["resid_post", 6][:, :]
probe_out = einsum(
    "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
    resid_post,
    linear_probe,
)
# print(probe_out.shape)

probe_log_probs = probe_out.log_softmax(-1)
probe_correct_log_probs = einops.reduce(
    probe_log_probs * state_stack_one_hot.cuda(),
    "modes batch pos rows cols options -> modes pos rows cols",
    "mean"
) * 3 # * 3 to correct for the mean
loss_even = -probe_correct_log_probs[0, 0::2].mean(0).sum()
loss_odd = -probe_correct_log_probs[1, 1::2].mean(0).sum()
loss_all = -probe_correct_log_probs[2, :].mean(0).sum()
# %%
torch.set_grad_enabled(False)
# %%
# Qualitative plot
# Note that mode=0 is even, mode=1 is odd, mode=2 is all
# Note further that we had an offset of 5, so mode=0 is odd, mode=1 is even, in practice, lol 
imshow(torch.stack([torch.stack([probe_out[0, 0, j].argmax(-1), state_stack_one_hot[0, j].argmax(-1).cuda()]) for j in range(5, 13)]), facet_col=1, animation_frame=0)
# %%


probe_accs = probe_log_probs.argmax(-1) == state_stack_one_hot.cuda().argmax(-1)
probe_accs = probe_accs.float().mean(1)
even_accs = probe_accs[0, pos_start:pos_end:2].mean(0)
odd_accs = probe_accs[1, pos_start+1:pos_end:2].mean(0)
all_accs = probe_accs[2, pos_start:pos_end].mean(0)
imshow(torch.stack([even_accs, odd_accs, all_accs]), facet_col=0, title="Accuracy Per Cell (Mean of 100 Games, Move 5-55)", facet_labels=["Odd", "Even", "All"])
even_accs_full = probe_accs[0, pos_start:pos_end].mean(0) 
odd_accs_full = probe_accs[1, pos_start:pos_end].mean(0) 
imshow(torch.stack([even_accs_full, odd_accs_full, all_accs]), facet_col=0, title="Accuracy Per Cell (Mean of 100 Games, Move 5-55)", facet_labels=["Odd (Full)", "Even (Full)", "All"])
# %%
probe_flipped_odd = probe_log_probs.clone()
probe_flipped_odd[..., ::2, :, :, 1] = probe_log_probs[..., ::2, :, :, 2]
probe_flipped_odd[..., ::2, :, :, 2] = probe_log_probs[..., ::2, :, :, 1]
probe_flipped_odd_accs = probe_flipped_odd.argmax(-1) == state_stack_one_hot.cuda().argmax(-1)
probe_flipped_even = probe_log_probs.clone()
probe_flipped_even[..., 1::2, :, :, 1] = probe_log_probs[..., 1::2, :, :, 2]
probe_flipped_even[..., 1::2, :, :, 2] = probe_log_probs[..., 1::2, :, :, 1]
probe_flipped_even_accs = probe_flipped_even.argmax(-1) == state_stack_one_hot.cuda().argmax(-1)
even_accs_flipped = probe_accs[0, pos_start:pos_end:2].mean(0) + probe_flipped_odd_accs.float().mean(1)[0, pos_start+1:pos_end:2].mean(0)
odd_accs_flipped = probe_accs[1, pos_start+1:pos_end:2].mean(0)+ probe_flipped_even_accs.float().mean(1)[1, pos_start:pos_end:2].mean(0)
all_accs = probe_accs[2, pos_start:pos_end].mean(0)
imshow(torch.stack([even_accs_flipped/2, odd_accs_flipped/2, all_accs]), facet_col=0, title="Flipped Accuracy Per Cell (Mean of 100 Games, Move 5-55)", facet_labels=["Odd (Flipped)", "Even (Flipped)", "All"])

# %%
imshow(torch.stack([even_accs, odd_accs, even_accs_flipped/2, odd_accs_flipped/2, all_accs]), facet_col=0, title="Accuracy Per Cell (Mean of 100 Games, Move 5-55)", facet_labels=["Odd", "Even", "Odd (All, Flipped)", "Even (All, Flipped)", "Baseline (All)"], color_continuous_scale="Blues", zmax=1, zmin=0.75)

# %%
mlp_acts = cache["post", 5][:, :, 1393]
alternating2 = torch.tensor([1 if i%2 == 0 else -1 for i in range(59)], device="cuda")
state_stack_flipped = state_stack * alternating2.cpu()[None, :, None, None]
indices = mlp_acts.flatten() > mlp_acts.flatten().quantile(0.99)
imshow(state_stack_flipped.reshape(-1, 8, 8).cuda()[indices].mean(0))
# %%
board_labels = [f"{i}{j}" for i in "abcdefgh".upper() for j in "01234567"]
grid = linear_probe[0, :, :, :, 2] - linear_probe[0, :, :, :, 1]
grid = grid.reshape(512, 64)
grid = grid/grid.norm(dim=0)
grid2 = linear_probe[1, :, :, :, 2] - linear_probe[1, :, :, :, 1]
grid2 = grid2.reshape(512, 64)
grid2 = grid2/grid2.norm(dim=0)
grid3 = torch.cat([grid, grid2], 1)
imshow(grid3.T @ grid3, x = [f"{lab}{k}" for k in ["(O)", "(E)"]for lab in board_labels], y = [f"{lab}{k}" for k in ["(O)", "(E)"]for lab in board_labels], title="Cosine Sim of B-W Linear Probe Directions by Cell (Odd Probe)", aspect="equal")
# %%
even_probe_dirs = linear_probe[1, :, :, :, 2] - linear_probe[1, :, :, :, 1]
# %%
pos = 20
plot_single_board(games_str[0, :pos+1])
# %%
# stack, labels = cache.get_full_resid_decomposition(-1, expand_neurons=False, pos_slice=pos,return_labels=True)
stack, labels = cache.decompose_resid(-1, pos_slice=pos,return_labels=True)
print(stack.shape)
stack = stack[:, 0]
indices = torch.tensor((state_stack[0, pos]==1).flatten()).cuda()
imshow(einsum("components d_model, d_model row col -> components row col", stack, even_probe_dirs).reshape(-1, 64)[:, indices], y=labels, x=[lab for c, lab in enumerate(board_labels) if indices[c].item()], title="Contributions to probe for Black Squares")
line(einsum("components d_model, d_model row col -> components row col", stack, even_probe_dirs).reshape(-1, 64)[:, indices].T, x=np.array(labels), line_labels=[lab for c, lab in enumerate(board_labels) if indices[c].item()], title="Contributions to probe for Black Squares")

indices = torch.tensor((state_stack[0, pos]==-1).flatten()).cuda()
imshow(einsum("components d_model, d_model row col -> components row col", stack, even_probe_dirs).reshape(-1, 64)[:, indices], y=labels, x=[lab for c, lab in enumerate(board_labels) if indices[c].item()], title="Contributions to probe for White Squares")
line(einsum("components d_model, d_model row col -> components row col", stack, even_probe_dirs).reshape(-1, 64)[:, indices].T, x=np.array(labels), line_labels=[lab for c, lab in enumerate(board_labels) if indices[c].item()], title="Contributions to probe for White Squares")
# %%
blank_probe = linear_probe[0, :, :, :, 0] - (linear_probe[0, :, :, :, 1] + linear_probe[0, :, :, :, 2])/2
black_probe = linear_probe[0, :, :, :, 2] - linear_probe[0, :, :, :, 1]
ni = 1393
x = torch.stack([
    (model.blocks[5].mlp.W_in[:, ni][:, None, None] * blank_probe).sum(0),
    (model.blocks[5].mlp.W_in[:, ni][:, None, None] * black_probe).sum(0),
    ], dim=2)
print(x.shape)
imshow(x, y=[i for i in "ABCDEFGH"], facet_col=2, facet_labels=["Blank", "Black - White"], zmin=-2, zmax=2, title=f"Input Weights for Neuron L5N{ni} via the probe")
# %%
for ni in [876, 1016, 561, 1379, 411]:
    x = torch.stack([
        (model.blocks[5].mlp.W_in[:, ni][:, None, None]/model.blocks[5].mlp.W_in[:, ni].norm() * blank_probe/blank_probe.norm(0, keepdim=True)).sum(0),
        (model.blocks[5].mlp.W_in[:, ni][:, None, None]/model.blocks[5].mlp.W_in[:, ni].norm() * black_probe/black_probe.norm(0, keepdim=True)).sum(0),
        ], dim=2)
    print(x.shape)
    imshow(x, y=[i for i in "ABCDEFGH"], facet_col=2, facet_labels=["Blank", "Black - White"], title=f"Input Weights for Neuron L5N{ni} via the probe")
# %%
pos = 20
plot_single_board(games_str[0, :pos+1])
state = torch.zeros((64,), dtype=torch.float32, device="cuda") - 10.
state[stoi_indices] = logits[0, pos].log_softmax(dim=-1)[1:]
imshow_pos(state.reshape(8, 8), zmax=0, zmin=-6, title="Logits at move 20", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal")
# %%
cell_r = 6
cell_c = 5
flip_dir = linear_probe[0, :, cell_r, cell_c, 2] - linear_probe[0, :, cell_r, cell_c, 1]
print(cache["resid_post", 6][0, pos] @ flip_dir)
print(cache["resid_post", 6][0, pos] @ linear_probe[0, :, cell_r, cell_c])
# %%
scale = 2
def flip_hook(resid, hook):
    coeff = resid[0, pos] @ flip_dir/flip_dir.norm()
    # if coeff.item() > 0:
    resid[0, pos] -= scale * coeff * flip_dir/flip_dir.norm()
flipped_logits = model.run_with_hooks(games_int[0:1, :pos+1],
                     fwd_hooks=[
                    #  ("blocks.3.hook_resid_post", flip_hook),
                     ("blocks.4.hook_resid_post", flip_hook),
                    #  ("blocks.5.hook_resid_post", flip_hook),
                    #  ("blocks.6.hook_resid_post", flip_hook),
                    #  ("blocks.7.hook_resid_post", flip_hook),
                     ]
                     ).log_softmax(dim=-1)[0, pos]
flip_state = torch.zeros((64,), dtype=torch.float32, device="cuda") - 10.
flip_state[stoi_indices] = flipped_logits.log_softmax(dim=-1)[1:]
# imshow_pos(state.reshape(8, 8), zmax=0, zmin=-6, title="Logits at move 20", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal")
imshow_pos(torch.stack([state.reshape(8, 8), flip_state.reshape(8, 8)]), zmax=0, zmin=-6, title="Logits pre and post intervening", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal", facet_col=0, facet_labels=["Original", "Flipped"])
# %%
pos = 50
game_index = 8
plot_single_board(games_str[game_index, :pos+1])
state = torch.zeros((64,), dtype=torch.float32, device="cuda") - 10.
state[stoi_indices] = logits[game_index, pos].log_softmax(dim=-1)[1:]
# imshow_pos(state.reshape(8, 8), zmax=0, zmin=-6, title="Logits at move 20", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal")
# %%
cell_r = 7
cell_c = 4
newly_legal = ["H7"]
newly_illegal = []
scale = 4
flip_dir = linear_probe[0, :, cell_r, cell_c, 2] - linear_probe[0, :, cell_r, cell_c, 1]
print(cache["resid_post", 6][game_index, pos] @ flip_dir)
print(cache["resid_post", 6][game_index, pos] @ linear_probe[0, :, cell_r, cell_c])
# %%
big_flipped_states_list = []
layer = 3
scales = [0, 1, 2, 4, 8, 16]
for scale in scales:
    def flip_hook(resid, hook):
        coeff = resid[0, pos] @ flip_dir/flip_dir.norm()
        # if coeff.item() > 0:
        resid[0, pos] -= (scale+1) * coeff * flip_dir/flip_dir.norm()
    flipped_logits = model.run_with_hooks(games_int[game_index:game_index+1, :pos+1],
                        fwd_hooks=[
                        #  ("blocks.3.hook_resid_post", flip_hook),
                        (f"blocks.{layer}.hook_resid_post", flip_hook),
                        #  ("blocks.5.hook_resid_post", flip_hook),
                        #  ("blocks.6.hook_resid_post", flip_hook),
                        #  ("blocks.7.hook_resid_post", flip_hook),
                        ]
                        ).log_softmax(dim=-1)[0, pos]
    flip_state = torch.zeros((64,), dtype=torch.float32, device="cuda") - 10.
    flip_state[stoi_indices] = flipped_logits.log_softmax(dim=-1)[1:]
    # imshow_pos(state.reshape(8, 8), zmax=0, zmin=-6, title="Logits at move 20", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal")
    # imshow_pos(torch.stack([state.reshape(8, 8), flip_state.reshape(8, 8)]), zmax=0, zmin=-6, title=f"Logits pre and post intervening at Layer {layer}", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal", facet_col=0, facet_labels=["Original", "Flipped"])
    # scatter(y=state, x=flip_state, title=f"Layer {layer} - Original vs Flipped", xaxis="Flipped", yaxis="Original", hover=board_labels)
    big_flipped_states_list.append(flip_state)
flip_state_big = torch.stack(big_flipped_states_list)
state_big = einops.repeat(state, "d -> b d", b=6)
color = torch.zeros((len(scales), 64)).cuda() + 0.2
for s in newly_legal:
    color[:, to_string(s)] = 1
for s in newly_illegal:
    color[:, to_string(s)] = -1
scatter(y=state_big, x=flip_state_big, title=f"Original vs Flipped {str_to_label(8*cell_r+cell_c)} at Layer {layer}", xaxis="Flipped", yaxis="Original", hover=board_labels, facet_col=0, facet_labels=[f"Translate by {i}x" for i in scales], color=color, color_name="Newly Legal", color_continuous_scale="Geyser")


# %%
cell_r = 5
cell_c = 5
cell_r_2 = 5
cell_c_2 = 6
newly_legal = []
newly_illegal = ["G6", "G7"]
scale = 3
flip_dir = linear_probe[0, :, cell_r, cell_c, 2] - linear_probe[0, :, cell_r, cell_c, 1]
flip_dir_2 = linear_probe[0, :, cell_r_2, cell_c_2, 2] - linear_probe[0, :, cell_r_2, cell_c_2, 1]
print(cache["resid_post", 6][game_index, pos] @ flip_dir)
print(cache["resid_post", 6][game_index, pos] @ linear_probe[0, :, cell_r, cell_c])
# %%
big_flipped_states_list = []
layer = 3
scales = [0, 1, 2, 4, 8, 16]
for scale in scales:
    def flip_hook(resid, hook):
        coeff = resid[0, pos] @ flip_dir/flip_dir.norm()
        # if coeff.item() > 0:
        resid[0, pos] -= (scale+1) * coeff * flip_dir/flip_dir.norm()
    def flip_hook_2(resid, hook):
        coeff = resid[0, pos] @ flip_dir_2/flip_dir_2.norm()
        # if coeff.item() > 0:
        resid[0, pos] -= (scale+1) * coeff * flip_dir_2/flip_dir_2.norm()
    flipped_logits = model.run_with_hooks(games_int[game_index:game_index+1, :pos+1],
                        fwd_hooks=[
                        #  ("blocks.3.hook_resid_post", flip_hook),
                        (f"blocks.{layer}.hook_resid_post", flip_hook),
                        (f"blocks.{layer}.hook_resid_post", flip_hook_2),
                        #  ("blocks.5.hook_resid_post", flip_hook),
                        #  ("blocks.6.hook_resid_post", flip_hook),
                        #  ("blocks.7.hook_resid_post", flip_hook),
                        ]
                        ).log_softmax(dim=-1)[0, pos]
    flip_state = torch.zeros((64,), dtype=torch.float32, device="cuda") - 10.
    flip_state[stoi_indices] = flipped_logits.log_softmax(dim=-1)[1:]
    # imshow_pos(state.reshape(8, 8), zmax=0, zmin=-6, title="Logits at move 20", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal")
    # imshow_pos(torch.stack([state.reshape(8, 8), flip_state.reshape(8, 8)]), zmax=0, zmin=-6, title=f"Logits pre and post intervening at Layer {layer}", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal", facet_col=0, facet_labels=["Original", "Flipped"])
    # scatter(y=state, x=flip_state, title=f"Layer {layer} - Original vs Flipped", xaxis="Flipped", yaxis="Original", hover=board_labels)
    big_flipped_states_list.append(flip_state)
flip_state_big = torch.stack(big_flipped_states_list)
state_big = einops.repeat(state, "d -> b d", b=6)
color = torch.zeros((len(scales), 64)).cuda() + 0.2
for s in newly_legal:
    color[:, to_string(s)] = 1
for s in newly_illegal:
    color[:, to_string(s)] = -1

scatter(y=state_big, x=flip_state_big, title=f"Original vs Flipped {str_to_label(8*cell_r+cell_c)} & {str_to_label(8*cell_r_2+cell_c_2)} at Layer {layer}", xaxis="Flipped", yaxis="Original", hover=board_labels, facet_col=0, facet_labels=[f"Translate by {i}x" for i in scales], color=color, color_name="Newly Legal", color_continuous_scale="Geyser")

# %%
linear_probe_old = torch.load("linear_probe_epoch1.pth")
linear_probe = torch.load("/workspace/othello_world/linear_probe_L4_blank_vs_color_v1.pth")
mode = 0
blank_probe_old = linear_probe_old[mode, :, :, :, 0] - linear_probe_old[mode, :, :, :, 1] * 0.5 - linear_probe_old[mode, :, :, :, 2] * 0.5
next_probe_old = linear_probe_old[mode, :, :, :, 2] - linear_probe_old[mode, :, :, :, 1]

blank_probe = linear_probe[0, :, :, :, 1] - linear_probe[0, :, :, :, 0]
next_probe = linear_probe[1, :, :, :, 1] - linear_probe[1, :, :, :, 0]

print(blank_probe.shape, blank_probe_old.shape)
print(next_probe.shape, next_probe_old.shape)

blank_probe = blank_probe / blank_probe.norm(dim=0, keepdim=True)
blank_probe = - blank_probe
blank_probe[:, 3:5, 3:5] = 0.
next_probe =  - next_probe / next_probe.norm(dim=0, keepdim=True)
# %%
imshow((blank_probe_old / blank_probe_old.norm(dim=0, keepdim=True)).reshape(512, 64).T @ (blank_probe / blank_probe.norm(dim=0, keepdim=True)).reshape(512, 64), title="Old Blank Probe", x=board_labels, y=board_labels)
imshow((next_probe_old / next_probe_old.norm(dim=0, keepdim=True)).reshape(512, 64).T @ (next_probe / next_probe.norm(dim=0, keepdim=True)).reshape(512, 64), title="Old Blank Probe", x=board_labels, y=board_labels)

# %%
imshow((blank_probe / blank_probe.norm(dim=0, keepdim=True)).reshape(512, 64).T @ (blank_probe / blank_probe.norm(dim=0, keepdim=True)).reshape(512, 64), title="Old Blank Probe", x=board_labels, y=board_labels)
imshow((next_probe / next_probe.norm(dim=0, keepdim=True)).reshape(512, 64).T @ (next_probe / next_probe.norm(dim=0, keepdim=True)).reshape(512, 64), title="Old Blank Probe", x=board_labels, y=board_labels)

# %%
moves_int = board_seqs_int[0, :30]
moves = to_label(moves_int)
print(moves)
plot_single_board(moves)
logits, cache = model.run_with_cache(moves_int)
imshow((cache["resid_post", 4][0, -1, :, None, None] * blank_probe).sum(0))
imshow((cache["resid_post", 4][0, -1, :, None, None] * next_probe).sum(0))
# %%
def patching_hook(resid_post, hook, row, col, scale=1):
    resid_post[0, -1, :] -= (resid_post[0, -1, :] @ next_probe[:, row, col]) * next_probe[:, row, col] * (scale + 1)
    return resid_post

# %%
offset = 4123456
games_int = board_seqs_int[offset:offset+50, :]
games_str = board_seqs_string[offset:offset+50, :]
logits, cache = model.run_with_cache(games_int[:, :-1])
# %%
pos = 30
game_index = 0
moves = games_str[game_index, :pos+1]
plot_single_board(games_str[game_index, :pos+1])
state = torch.zeros((64,), dtype=torch.float32, device="cuda") - 10.
state[stoi_indices] = logits[game_index, pos].log_softmax(dim=-1)[1:]
imshow_pos(state.reshape(8, 8), zmax=0, zmin=-6, title="Logits at move 20", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal")
# %%



cell_r = 2
cell_c = 5
# newly_legal = ["F1", "A6"]
# newly_illegal = []
scale = 4
flip_dir = next_probe[:, cell_r, cell_c]
print(cache["resid_post", 4][game_index, pos] @ flip_dir)
print(cache["resid_post", 4][game_index, pos] @ blank_probe[:, cell_r, cell_c])

board = OthelloBoardState()
board.update(moves.tolist())
board_state = board.state.copy()
valid_moves = board.get_valid_moves()
flipped_board = copy.deepcopy(board)
flipped_board.state[cell_r, cell_c] *= -1
flipped_valid_moves = flipped_board.get_valid_moves()

newly_legal = [str_to_label(move) for move in flipped_valid_moves if move not in valid_moves]
newly_illegal = [str_to_label(move) for move in valid_moves if move not in flipped_valid_moves]
print("newly_legal", newly_legal)
print("newly_illegal", newly_illegal)

imshow([board.state, flipped_board.state], facet_col=0)

# %%
big_flipped_states_list = []
layer = 3
scales = [0, 1, 2, 4, 8, 16]
for scale in scales:
    def flip_hook(resid, hook):
        coeff = resid[0, pos] @ flip_dir/flip_dir.norm()
        # if coeff.item() > 0:
        resid[0, pos] -= (scale+1) * coeff * flip_dir/flip_dir.norm()
    flipped_logits = model.run_with_hooks(games_int[game_index:game_index+1, :pos+1],
                        fwd_hooks=[
                        #  ("blocks.3.hook_resid_post", flip_hook),
                        (f"blocks.{layer}.hook_resid_post", flip_hook),
                        #  ("blocks.5.hook_resid_post", flip_hook),
                        #  ("blocks.6.hook_resid_post", flip_hook),
                        #  ("blocks.7.hook_resid_post", flip_hook),
                        ]
                        ).log_softmax(dim=-1)[0, pos]
    flip_state = torch.zeros((64,), dtype=torch.float32, device="cuda") - 10.
    flip_state[stoi_indices] = flipped_logits.log_softmax(dim=-1)[1:]
    # imshow_pos(state.reshape(8, 8), zmax=0, zmin=-6, title="Logits at move 20", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal")
    # imshow_pos(torch.stack([state.reshape(8, 8), flip_state.reshape(8, 8)]), zmax=0, zmin=-6, title=f"Logits pre and post intervening at Layer {layer}", y=[i for i in "ABCDEFGH"], x=[i for i in "01234567"], aspect="equal", facet_col=0, facet_labels=["Original", "Flipped"])
    # scatter(y=state, x=flip_state, title=f"Layer {layer} - Original vs Flipped", xaxis="Flipped", yaxis="Original", hover=board_labels)
    big_flipped_states_list.append(flip_state)
flip_state_big = torch.stack(big_flipped_states_list)
state_big = einops.repeat(state, "d -> b d", b=6)
color = torch.zeros((len(scales), 64)).cuda() + 0.2
for s in newly_legal:
    color[:, to_string(s)] = 1
for s in newly_illegal:
    color[:, to_string(s)] = -1
scatter(y=state_big, x=flip_state_big, title=f"Original vs Flipped {str_to_label(8*cell_r+cell_c)} at Layer {layer}", xaxis="Flipped", yaxis="Original", hover=board_labels, facet_col=0, facet_labels=[f"Translate by {i}x" for i in scales], color=color, color_name="Newly Legal", color_continuous_scale="Geyser")
# %%
# def one_hot(list_of_ints, num_classes=64):
#     out = torch.zeros((num_classes,), dtype=torch.float32)
#     out[list_of_ints] = 1.
#     return out
# offset = 4123456
# num_games = 2000
# games_int = board_seqs_int[offset:offset+num_games]
# games_str = board_seqs_string[offset:offset+num_games]
# states = np.zeros((num_games, 8, 8), dtype=np.float32)
# valid_moves = torch.zeros((num_games, 64), dtype=torch.float32)
# for i in tqdm(range(num_games)):
#     board = OthelloBoardState()
#     board.update(games_str[i].tolist())
#     states[i] = board.state
#     valid_moves[i] = one_hot(board.get_valid_moves())
# # %%
# logits, cache = model.run_with_cache(games_int[:, :-1].cuda())
# # %%
# othello_backup_dict = {
#     "cache": cache,
#     "logits": logits,
#     "games_int": games_int,
#     "games_str": games_str,
#     "offset": offset,
#     "states": states,
#     "valid_moves": valid_moves,
# }

# torch.save(othello_backup_dict, "/workspace/othello_world/othello_backup_dict.pt")
# %%
imshow(big_cache["pattern", 0][0, :], facet_col=0)
# %%
U_next, S_next, Vh_next = next_probe.reshape(512, 64).svd()
U_blank, S_blank, Vh_blank = blank_probe.reshape(512, 64).svd()
U_embed, S_embed, Vh_embed = model.W_E.T.svd()
U_unembed, S_unembed, Vh_unembed = model.W_U.svd()
U_probe, S_probe, Vh_probe = torch.cat([blank_probe, next_probe], dim=1).reshape(512, 128).svd()
U_blank = U_blank[:, :-4]
U_probe = U_probe[:, :-4]

# %%
line(S_blank)
line(S_next)
line(S_probe)
# %%

out = []
out_labels = []
for label, U in [("blank", U_blank), ("next", U_next), ("blank+next", U_probe)]:
    for act_name in ["attn_out", "mlp_out", "resid_pre"]:
        frac_retained = torch.stack([(((big_cache[act_name, layer])) @ U).norm(dim=-1) / ((big_cache[act_name, layer])).norm(dim=-1) for layer in range(8)]).pow(2)[:, :, 10:-10].mean(dim=[1, 2])
        out.append(frac_retained)
        # line(frac_retained[:, :, 30], title="Fracttion of Residual Captured by Blank + Next", xaxis="Layer", yaxis="Fraction Captured")
        out_labels.append(f"{label} {act_name}")

imshow(out, yaxis="Mode", xaxis="Layer", title="Average Fraction of Residual Captured by Blank + Next", color_continuous_midpoint=0.125, zmax=1, zmin=-0.75, y=out_labels)
# %%
# imshow(model.W_E @ U_probe, title="Embed to Blank + Next", yaxis="Embedding", xaxis="Blank + Next", color_continuous_midpoint=0, zmax=1, zmin=-1)

fig = line([(model.W_E @ U).pow(2).sum(dim=-1) / model.W_E.pow(2).sum(dim=-1) for U in [U_blank, U_next, U_probe]], return_fig=True, line_labels = ["blank", "next", "blank + next"], title="Embed vs Probe Subspaces", range_y=(0, 1))
fig.add_hline(y=0.25)
fig.show()
fig = line([(model.W_U.T @ U).pow(2).sum(dim=-1) / model.W_U.T.pow(2).sum(dim=-1) for U in [U_blank, U_next, U_probe]], return_fig=True, line_labels = ["blank", "next", "blank + next"], title="Unembed vs Probe Subspaces", range_y=(0, 1))
fig.add_hline(y=0.25)
fig.show()
# %%
Us = [("next", U_next), ("blank", U_blank), ("embed", U_embed)]
line_list = []
label_list = []
for layer in range(8):
    W = model.blocks[layer].mlp.W_in.T
    for label, U in Us:
        line_list.append((W @ U).pow(2).sum(dim=-1) / W.pow(2).sum(dim=-1))
        label_list.append(f"{label} {layer}")

fig = line(line_list, return_fig=True, line_labels = label_list, title="W_in vs Probe Subspaces", range_y=(0, 1))
fig.add_hline(y=0.125)
fig.show()
# %%
Us = [("next", U_next), ("blank", U_blank), ("embed", U_embed)]
line_list = []
label_list = []
for layer in range(8):
    W = model.blocks[layer].mlp.W_in.T
    for label, U in Us:
        line_list.append((W @ U).pow(2).sum(dim=-1) / W.pow(2).sum(dim=-1))
        label_list.append(f"{label} {layer}")

fig = line([lis.sort().values for lis in line_list], return_fig=True, line_labels = label_list, title="W_in vs Probe Subspaces", range_y=(0, 1))
fig.add_hline(y=0.125)
fig.show()

# %%
def state_stack_to_one_hot(state_stack):
    one_hot = torch.zeros(
        2, # blank vs color (mode)
        state_stack.shape[0],
        state_stack.shape[1],
        8, # rows
        8, # cols
        2, # the two options
        device=state_stack.device,
        dtype=torch.int,
    )
    one_hot[0, ..., 0] = state_stack == 0
    one_hot[0, ..., 1] = 1 - one_hot[0, ..., 0]
    one_hot[1, :, 0::2, :, :, 0] = (state_stack == 1)[:, 0::2] # black to play because we start on move 3
    one_hot[1, :, 1::2, :, :, 0] = (state_stack == -1)[:, 1::2]
    one_hot[1, ..., 1] = 1 - one_hot[1, ..., 0]
    return one_hot
# big_states = torch.tensor(big_states)
big_state_stack_one_hot = state_stack_to_one_hot(big_states)
resid_post = big_cache["resid_post", 4]
linear_probe = torch.load("/workspace/othello_world/linear_probe_L4_blank_vs_color_v1.pth")
probe_out = einsum(
    "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
    resid_post,
    linear_probe,
).cpu()
next_probe_out = einsum(
    "batch pos d_model, d_model rows cols -> batch pos rows cols",
    resid_post,
    next_probe,
).cpu()
# print(probe_out.shape)

acc_blank = (probe_out[0].argmax(-1) == big_state_stack_one_hot[0].argmax(-1)).float().mean()
acc_color = (((next_probe_out[:, 5:-5]>0).int() == big_state_stack_one_hot[1,:, 5:-5].argmax(-1)) * big_state_stack_one_hot[1,:, 5:-5].sum(-1)).float().sum()/(big_state_stack_one_hot[1,:, 5:-5]).float().sum()
print(acc_blank)
print(acc_color)
# %%
imshow([big_states[50, 30] * (big_state_stack_one_hot[1, 50, 30].argmax(-1) - 0.5), big_states[50, 30] * (probe_out[1, 50, 30].argmax(-1) - 0.5)], facet_col=0)
imshow([(next_probe_out[50, 30]>0).int() * (big_state_stack_one_hot[1, 50, 30].argmax(-1) - 0.5), big_states[50, 30] * (probe_out[1, 50, 30].argmax(-1) - 0.5)], facet_col=0)
imshow([big_states[50, 31] * (big_state_stack_one_hot[1, 50, 31].argmax(-1) - 0.5), big_states[50, 31] * (probe_out[1, 50, 31].argmax(-1) - 0.5)], facet_col=0)

# %%
def get_acc_next(tensor):
    assert tensor.shape == (2000, 59, 512)
    next_probe_out = einsum(
        "batch pos d_model, d_model rows cols -> batch pos rows cols",
        tensor,
        next_probe,
    ).cpu()
    return 0.5 * ((((next_probe_out>0)[:, 2:-2:2] & (big_states==-1)[:, 2:-2:2]).sum() + 
    ((next_probe_out<0)[:, 2:-2:2] & (big_states==1)[:, 2:-2:2]).sum()) / (big_states!=0)[:, 2:-2:2].sum() +
    (((next_probe_out>0)[:, 3:-2:2] & (big_states==1)[:, 3:-2:2]).sum() + 
    ((next_probe_out<0)[:, 3:-2:2] & (big_states==-1)[:, 3:-2:2]).sum()) / (big_states!=0)[:, 3:-2:2].sum())
def get_acc_blank(tensor):
    assert tensor.shape == (2000, 59, 512)
    blank_probe_out = einsum(
        "batch pos d_model, d_model rows cols -> batch pos rows cols",
        tensor,
        blank_probe,
    ).cpu()
    # imshow(blank_probe_out[50, 30])
    # imshow(big_states[50, 30])
    return ((blank_probe_out<0)[:, 2:-2] & (big_states==0)[:, 2:-2]).float().mean() + ((blank_probe_out>=0)[:, 2:-2] & (big_states!=0)[:, 2:-2]).float().mean()
print(get_acc_color(big_cache["resid_post", 4]))
print(get_acc_blank(big_cache["resid_post", 4]))
# %%
line([[get_acc_next(big_cache[act_name, layer]) for layer in range(8)] for act_name in ["attn_out", "mlp_out", "resid_post"]], title="Accuracy of Next Move Prediction", line_labels=["attn_out", "mlp_out", "resid_post"], range_y=(0., 1))
line([[get_acc_blank(big_cache[act_name, layer]) for layer in range(8)] for act_name in ["attn_out", "mlp_out", "resid_post"]], title="Accuracy of Blank Move Prediction", line_labels=["attn_out", "mlp_out", "resid_post"], range_y=(0., 1))
# %%
line([[get_acc_blank(big_cache["z",layer][:, :, head] @ model.blocks[layer].attn.W_O[head]) for head in range(8)] for layer in range(8)], title="Accuracy of Blank Move Prediction per layer 0 head", range_y=(0., 1))

# %%
layer = 5
ni = 1393
vec = model.W_in[layer, :, ni]
vec /= vec.norm()
line([vec @ blank_probe.reshape(512, 64), vec @ next_probe.reshape(512, 64)], x=np.array(board_labels), title="Blank and Next Probe for Layer 5, Neuron 1393")
# %%
W_in = einops.rearrange(model.W_in, "layer d_model neuron -> (layer neuron) d_model")
neuron_weights_shifted = W_in @ torch.cat([blank_probe.reshape(512, 64), next_probe.reshape(512, 64)], dim=1)
neuron_labels = [f"L{l}N{n}" for l in range(8) for n in range(model.cfg.d_mlp)]

line(neuron_weights_shifted[:, 98], title="Neuron Input Weights for E2 Probe")
# %%
def plot_neuron(ni, layer=0):
    if ni>2048:
        layer = ni//2048
        ni = ni % 2048
    vec = model.W_in[layer, :, ni]
    vec /= vec.norm()
    
    blank_board = (vec[:, None, None] * blank_probe).sum(0)
    next_board = (vec[:, None, None] * next_probe).sum(0)
    imshow([blank_board, next_board], facet_col=0, title=f"Blank and Next Probe for Layer {layer}, Neuron {ni}", facet_labels=["Blank", "Next"])
    
plot_neuron(ni, layer)
# %%
next_probe_old_norm = next_probe_old / next_probe_old.norm(dim=0, keepdim=True)
imshow([(vec[:, None, None] * next_probe_old_norm).sum(0), (vec[:, None, None] * next_probe).sum(0)], facet_col=0)
blank_probe_old_norm = blank_probe_old / blank_probe_old.norm(dim=0, keepdim=True)
imshow([(vec[:, None, None] * blank_probe_old_norm).sum(0), (vec[:, None, None] * blank_probe).sum(0)], facet_col=0)
# %%
W_in = einops.rearrange(model.W_in, "layer d_model neuron -> (layer neuron) d_model")
W_in /= W_in.norm(dim=-1, keepdim=True)
neuron_weights_shifted = W_in @ torch.cat([blank_probe_old_norm.reshape(512, 64), next_probe_old_norm.reshape(512, 64)], dim=1)
neuron_labels = [f"L{l}N{n}" for l in range(8) for n in range(model.cfg.d_mlp)]

line(neuron_weights_shifted[:, 98], title="Neuron Input Weights for E2 Probe")

# %%
values, indices = neuron_weights_shifted.sort(dim=0, descending=True)
imshow(values[:100, :], hover=indices[:100, :], x=[f"{lab} {t}" for t in ["blank", "next"] for lab in board_labels])
imshow(values[-100:, :], hover=indices[-100:, :], x=[f"{lab} {t}" for t in ["blank", "next"] for lab in board_labels])
# %%
index = 9126
plot_neuron(index)
# %%
neuron_acts = big_cache["post", index//2048][:, :, index % 2048]
sorted_neuron_acts, top_board_indices = neuron_acts.flatten().sort(descending=True)
print(top_board_indices)
for b in top_board_indices[:5]:
    game_index = b // 59
    move_index = b % 59
    print(f"Game {game_index}, Move {move_index}")
    # plot_single_board(games_str[game_index, :move_index])
# %%
cutoff = 200
top_board_states = big_states.reshape(-1, 8, 8)[top_board_indices[:cutoff]]
parity = (top_board_indices[:cutoff] % 59) % 2
parity = (parity * 2) - 1
print(parity)
abs_state_top = top_board_states.abs().mean(dim=0)
alt_state_top = (top_board_states * parity[:, None, None].cpu()).mean(dim=0)
line(sorted_neuron_acts[:cutoff], title=f"Top {cutoff} Neuron activations for L{index // 2048}N{index % 2048}")
imshow([abs_state_top, alt_state_top], facet_col=0, facet_labels=["Absolute", "Alternating"])
# %%
