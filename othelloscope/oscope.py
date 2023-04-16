# Import libraries for reading and writing files
import os
import sys

# Import libraries for image processing
import numpy as np

# Import stuff
import torch
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from neel_plotly import line, scatter, imshow, histogram
import einops

import transformer_lens.utils as utils
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
)

torch.set_grad_enabled(False)

import transformer_lens.utils as utils

OTHELLO_ROOT = Path(".")
# Import othello util functions from file mechanistic_interpretability/mech_interp_othello_utils.py
sys.path.append(str(OTHELLO_ROOT))
sys.path.append(str(OTHELLO_ROOT / "mechanistic_interpretability"))
from mech_interp_othello_utils import (
    plot_single_board,
    to_string,
    to_int,
    int_to_label,
    string_to_label,
    OthelloBoardState,
)

USE_CUDA: bool = torch.cuda.is_available()
DEVICE: str = "cuda" if USE_CUDA else "cpu"


def to_device(object):
    if USE_CUDA:
        return object.cuda()
    else:
        return object.cpu()


def generate_from_template(template, *args):
    """Generate a file from a template.

    Parameters
    ----------
    template : str
        Path to the template file.
    **kwargs
        Keyword arguments to be used in the template.

    Returns
    -------
    str
        The generated file.
    """
    with open(template, "r") as f:
        template = f.read()
    return template.format(*args)


def generate_neuron_path(layer, neuron):
    """Generate the path to the neuron.

    Parameters
    ----------
    layer : int
        The layer of the neuron.
    neuron : int
        The index of the neuron.

    Returns
    -------
    str
        The path to the neuron.
    """
    return "othelloscope/L{0}/N{1}".format(layer, neuron)


def generate_activation_table(heatmap):
    """Generate an activation table.

    Parameters
    ----------
    layer : int
        The layer of the neuron.
    neuron : int
        The index of the neuron.

    Returns
    -------
    str
        The generated activation table.
    """
    # Convert heatmap to numpy array
    heatmap = np.array(heatmap[-1].detach().cpu())
    othello_board = np.array(
        [
            ["A", "B", "C", "D", "E", "F", "G", "H"],
            ["1", "2", "3", "4", "5", "6", "7", "8"],
        ]
    )

    # Create a table
    table = "<table>"

    # Loop through the rows
    for row in range(heatmap.shape[0]):
        table += "<tr>"

        # Loop through the columns
        for col in range(heatmap.shape[1]):
            table += "<td title={0}>{1}</td>".format(
                heatmap[row, col], othello_board[0, row] + othello_board[1, col]
            )

        table += "</tr>"

    table += "</table>"

    return table


def one_hot(list_of_ints, num_classes=64):
    out = torch.zeros((num_classes,), dtype=torch.float32)
    out[list_of_ints] = 1.0
    return out


def generate_probability_table(
    layer, game_index, move, focus_cache, linear_probe, **kwargs
):
    """Generate a probability table."""
    residual_stream = to_device(focus_cache["resid_post", layer][game_index, move])

    print("residual_stream", residual_stream.shape)
    probe_out = einops.einsum(
        residual_stream,
        linear_probe,
        "d_model, d_model row col options -> row col options",
    )
    probabilities = probe_out.softmax(dim=-1)
    return probabilities


def state_stack_to_one_hot(state_stack):
    """Convert a state stack to one hot encoding.

    Parameters
    ----------
    state_stack : torch.Tensor
        The state stack.

    Returns
    -------
    torch.Tensor
        The one hot encoding.
    """
    one_hot = torch.zeros(
        state_stack.shape[0],  # num games
        state_stack.shape[1],  # num moves
        8,  # rows
        8,  # cols
        3,  # the two options
        device=state_stack.device,
        dtype=torch.int,
    )
    one_hot[..., 0] = state_stack == 0  # empty
    one_hot[..., 1] = state_stack == -1  # white
    one_hot[..., 2] = state_stack == 1  # black

    return one_hot


def neuron_probe(model, layer, neuron):
    neuron = neuron.item()
    w_out = model.blocks[layer].mlp.W_out[neuron, :].detach()
    w_out /= w_out.norm()
    return w_out


def layer_probe(
    model,
    layer,
    focus_cache,
    blank_probe_normalised,
    my_probe_normalised,
    stoi_indices,
    board_seqs_int,
    board_seqs_string,
):
    """Generate a layer probe, all the heatmaps, and the page.

    Parameters
    ----------
    model : HookedTransformer
        The model.
    layer : int
        The layer to generate the probe for.
    focus_cache : dict
        The focus cache.
    blank_probe_normalised : torch.Tensor
        The normalised blank probe.
    my_probe_normalised : torch.Tensor
        The normalised my probe.
    stoi_indices : range
        The stoi indices.

    Returns
    -------
    None
    """

    neurons = (
        to_device(focus_cache["post", layer][:, 3:-3])
        .std(dim=[0, 1])
        .argsort(descending=True)
    )
    heatmaps_blank = []
    heatmaps_my = []
    for idx, neuron in enumerate(neurons):
        w_out = neuron_probe(model, layer, neuron)
        heatmaps_blank.append(
            (w_out[:, None, None] * blank_probe_normalised).sum(dim=0)
        )
        heatmaps_my.append((w_out[:, None, None] * my_probe_normalised).sum(dim=0))
        games = top_50_games(
            layer, neuron, focus_cache, board_seqs_int, board_seqs_string
        )
        generate_page(
            layer, idx, "TEST1", heatmaps_blank, heatmaps_my, games, model, stoi_indices
        )


def generate_page(
    layer, neuron, rank, heatmaps_blank, heatmaps_my, games, model, stoi_indices
):
    """Generate a page."""

    # Get the path to the neuron
    path = generate_neuron_path(layer, neuron)

    # Create a folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Read the template file
    template = generate_from_template(
        "othelloscope/template.html",
        (
            f"<a href='../../L{layer}/N{neuron - 1}/index.html'>Previous neuron</a> - "
            if neuron > 0
            else (
                f"<a href='../../L{layer-1}/N{2047}'>Previous layer</a> - "
                if layer > 0
                else ""
            )
        )
        + (
            f"<a href='../../L{layer}/N{neuron + 1}/index.html'>Next</a>"
            if neuron < 2047
            else (f"<a href='../../L{layer+1}/N0'>Next layer</a>" if layer < 7 else "")
        ),
        layer,
        neuron,
        rank,
        generate_activation_table(heatmaps_blank),
        generate_activation_table(heatmaps_my),
        generate_logit_attribution_table(layer, neuron, model, stoi_indices),
        games,
    )

    # Write the generated file
    with open(path + "/index.html", "w") as f:
        f.write(template)


def top_50_games(layer, neuron, focus_cache, board_seqs_int, board_seqs_string):
    """Takes top 50 games and visualizes them in a grid with visualization of the board state when hovering along with the neuron activation for that specific game."""
    neuron_acts = focus_cache["post", layer, "mlp"][:, :, neuron]
    num_games = 50
    focus_games_int = board_seqs_int[:num_games]
    focus_games_string = board_seqs_string[:num_games]

    # Generate a table where y = game, x = move, and the value is the move from focus_games_string
    moves = []
    for game in focus_games_int:
        moves.append([int_to_label(move) for move in game])

    # Return HTML table with the title = activation of the neuron
    # Create a table
    table = "<table class='games'>"

    # Loop through the rows
    table += "<tr><td class='game_step_id'></td>"
    for col in range(neuron_acts.shape[1] + 1):
        table += "<td class='game_step_id'>{0}</td>".format(col + 1)
    table += "</tr>"
    for row in range(neuron_acts.shape[0]):
        table += "<tr>"
        table += "<td class='game_id'>{0}</td>".format(row + 1)

        # Loop through the columns
        for col in range(neuron_acts.shape[1]):
            table += "<td class='game_step' title={0}>{1}</td>".format(
                neuron_acts[row, col], moves[row][col]
            )

        table += "</tr>"

    table += "</table>"

    return table


def generate_logit_attribution_table(layer, neuron, model, stoi_indices):
    """Generate a logit attribution table."""
    w_out = model.blocks[layer].mlp.W_out[neuron, :]
    state = torch.zeros(8, 8, device="cpu")
    state.flatten()[stoi_indices] = w_out @ model.W_U[:, 1:]
    state.reshape(8, 8)
    return generate_activation_table([state])


def main():
    """Main function."""

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
    model = to_device(HookedTransformer(cfg))

    sd = utils.download_file_from_hf(
        "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
    )
    model.load_state_dict(sd)

    board_seqs_int = torch.tensor(
        np.load(OTHELLO_ROOT / "board_seqs_int_small.npy"), dtype=torch.long
    )
    board_seqs_string = torch.tensor(
        np.load(OTHELLO_ROOT / "board_seqs_string_small.npy"), dtype=torch.long
    )

    num_games, length_of_game = board_seqs_int.shape
    print(
        "Number of games:",
        num_games,
    )
    print("Length of game:", length_of_game)

    stoi_indices = list(range(0, 60))
    alpha = "ABCDEFGH"

    def to_board_label(i):
        return f"{alpha[i//8]}{i%8}"

    board_labels = list(map(to_board_label, stoi_indices))

    moves_int = board_seqs_int[0, :30]

    # This is implicitly converted to a batch of size 1
    logits = model(moves_int)
    print("logits:", logits.shape)

    logit_vec = logits[0, -1]
    log_probs = logit_vec.log_softmax(-1)
    # Remove passing
    log_probs = log_probs[1:]
    assert len(log_probs) == 60

    temp_board_state = torch.zeros(64, device=logit_vec.device)
    # Set all cells to -15 by default, for a very negative log prob - this means the middle cells don't show up as mattering
    temp_board_state -= 13.0
    temp_board_state[stoi_indices] = log_probs
    temp_board_state = temp_board_state.reshape(8, 8)
    temp_board_state = temp_board_state.cpu().detach().numpy()

    print("temp_board_state:", temp_board_state.shape)

    num_games = 50
    focus_games_int = board_seqs_int[:num_games]
    focus_games_string = board_seqs_string[:num_games]

    focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
    focus_valid_moves = torch.zeros((num_games, 60, 64), dtype=torch.float32)
    for i in range(num_games):
        board = OthelloBoardState()
        for j in range(60):
            board.umpire(focus_games_string[i, j].item())
            focus_states[i, j] = board.state
            focus_valid_moves[i, j] = one_hot(board.get_valid_moves())
    print("focus states:", focus_states.shape)
    print("focus_valid_moves", focus_valid_moves.shape)

    focus_logits, focus_cache = model.run_with_cache(to_device(focus_games_int[:, :-1]))

    full_linear_probe = torch.load(
        OTHELLO_ROOT / "main_linear_probe.pth", map_location=DEVICE
    )
    rows = 8

    cols = 8
    options = 3
    black_to_play_index = 0
    white_to_play_index = 1
    blank_index = 0
    their_index = 1
    my_index = 2
    linear_probe = torch.zeros(cfg.d_model, rows, cols, options, device=DEVICE)
    linear_probe[..., blank_index] = 0.5 * (
        full_linear_probe[black_to_play_index, ..., 0]
        + full_linear_probe[white_to_play_index, ..., 0]
    )
    linear_probe[..., their_index] = 0.5 * (
        full_linear_probe[black_to_play_index, ..., 1]
        + full_linear_probe[white_to_play_index, ..., 2]
    )
    linear_probe[..., my_index] = 0.5 * (
        full_linear_probe[black_to_play_index, ..., 2]
        + full_linear_probe[white_to_play_index, ..., 1]
    )

    layer = 6
    game_index = 1
    move = 22

    # plot_single_board(int_to_label(focus_games_int[game_index, : move + 1]))

    # This is the linear probe
    probabilities = generate_probability_table(
        layer,
        game_index,
        move,
        title="Linear probe",
        focus_cache=focus_cache,
        linear_probe=linear_probe,
    )
    print("probabilities:", probabilities.shape)

    print("Computing accuracy over 50 games")
    # We first convert the board states to be in terms of my (+1) and their (-1)
    alternating = np.array(
        [-1 if i % 2 == 0 else 1 for i in range(focus_games_int.shape[1])]
    )
    flipped_focus_states = focus_states * alternating[None, :, None, None]

    # We now convert to one hot
    focus_states_flipped_one_hot = state_stack_to_one_hot(
        to_device(torch.tensor(flipped_focus_states))
    )

    # Take the argmax
    focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)
    probe_out = einops.einsum(
        to_device(focus_cache["resid_post", 6]),
        linear_probe,
        "game move d_model, d_model row col options -> game move row col options",
    )
    probe_out_value = probe_out.argmax(dim=-1)

    correct_middle_odd_answers = (
        to_device(probe_out_value) == focus_states_flipped_value[:, :-1]
    )[:, 5:-5:2]
    accuracies_odd = einops.reduce(
        correct_middle_odd_answers.float(), "game move row col -> row col", "mean"
    )
    correct_middle_answers = (
        to_device(probe_out_value) == focus_states_flipped_value[:, :-1]
    )[:, 5:-5]
    accuracies = einops.reduce(
        correct_middle_answers.float(), "game move row col -> row col", "mean"
    )
    print("Accuracy over 50 games:", accuracies.mean().item())

    blank_probe = (
        linear_probe[..., 0] - linear_probe[..., 1] * 0.5 - linear_probe[..., 2] * 0.5
    )
    my_probe = linear_probe[..., 2] - linear_probe[..., 1]

    pos = 20
    game_index = 0
    moves = focus_games_string[game_index, : pos + 1]
    state = torch.zeros((64,), dtype=torch.float32, device=DEVICE) - 10.0
    state[stoi_indices] = focus_logits[game_index, pos].log_softmax(dim=-1)[1:]
    print("state:", state.shape)

    print("\nACTIVATE NEURON REPRESENTATION\n")
    # Scale the probes down to be unit norm per cell
    blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
    my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)
    # Set the center blank probes to 0, since they're never blank so the probe is meaningless
    blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

    layer = 5
    neuron = 1393
    w_in = model.blocks[layer].mlp.W_in[:, neuron].detach()
    w_in /= w_in.norm()
    w_out = model.blocks[layer].mlp.W_out[neuron, :].detach()
    w_out /= w_out.norm()

    U, S, Vh = torch.svd(
        torch.cat(
            [my_probe.reshape(cfg.d_model, 64), blank_probe.reshape(cfg.d_model, 64)],
            dim=1,
        )
    )
    # Remove the final four dimensions of U, as the 4 center cells are never blank and so the blank probe is meaningless there
    probe_space_basis = U[:, :-4]

    print(
        "Fraction of input weights in probe basis:",
        (w_in @ probe_space_basis).norm().item() ** 2,
    )
    print(
        "Fraction of output weights in probe basis:",
        (w_out @ probe_space_basis).norm().item() ** 2,
    )

    for layer in range(8):
        print(f"Layer {layer} converted")
        layer_probe(
            model,
            layer,
            focus_cache,
            blank_probe_normalised,
            my_probe_normalised,
            stoi_indices,
            board_seqs_int,
            board_seqs_string,
        )


# Make an 8x8 html table from a numpy array
# table = generate_activation_table(3, 1123)

# # Read the template file
# template = generate_from_template(
#     "othelloscope/index.html",
#     3,
#     1123,
#     "Doe",
# )

# # Create a folder if it doesn't exist
# if not os.path.exists("othelloscope/test"):
#     os.makedirs("othelloscope/test")

# # Write the generated file
# with open("othelloscope/test/test.html", "w") as f:
#     f.write(template)


if __name__ == "__main__":
    main()
