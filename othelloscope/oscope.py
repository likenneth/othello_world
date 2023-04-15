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


def get_neuron_at_layer(layer, neuron):
    """Get the neuron at a specific layer.

    Parameters
    ----------
    layer : int
        The layer of the neuron.
    neuron : int
        The index of the neuron.

    Returns
    -------
    str
        The neuron at the layer.
    """
    return "layer_{}_neuron_{}".format(layer, neuron)


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
    return "othelloscope/L{}/N{}".format(layer, neuron)


def generate_activation_table(layer, neuron):
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
    # Get the path to the neuron
    path = generate_neuron_path(layer, neuron)

    # Read the activation file
    activation = np.load("{}/activation.npy".format(path))

    # Create a table
    table = "<table>"

    # Loop through the rows
    for row in range(activation.shape[0]):
        table += "<tr>"

        # Loop through the columns
        for col in range(activation.shape[1]):
            table += "<td>{}</td>".format(activation[row, col])

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
    residual_stream = focus_cache["resid_post", layer][game_index, move]
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
    model = HookedTransformer(cfg)

    sd = utils.download_file_from_hf(
        "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
    )
    # champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
    model.load_state_dict(sd)

    # An example input
    sample_input = torch.tensor(
        [
            [
                20,
                19,
                18,
                10,
                2,
                1,
                27,
                3,
                41,
                42,
                34,
                12,
                4,
                40,
                11,
                29,
                43,
                13,
                48,
                56,
                33,
                39,
                22,
                44,
                24,
                5,
                46,
                6,
                32,
                36,
                51,
                58,
                52,
                60,
                21,
                53,
                26,
                31,
                37,
                9,
                25,
                38,
                23,
                50,
                45,
                17,
                47,
                28,
                35,
                30,
                54,
                16,
                59,
                49,
                57,
                14,
                15,
                55,
                7,
            ]
        ]
    )
    # The argmax of the output (ie the most likely next move from each position)
    sample_output = torch.tensor(
        [
            [
                21,
                41,
                40,
                34,
                40,
                41,
                3,
                11,
                21,
                43,
                40,
                21,
                28,
                50,
                33,
                50,
                33,
                5,
                33,
                5,
                52,
                46,
                14,
                46,
                14,
                47,
                38,
                57,
                36,
                50,
                38,
                15,
                28,
                26,
                28,
                59,
                50,
                28,
                14,
                28,
                28,
                28,
                28,
                45,
                28,
                35,
                15,
                14,
                30,
                59,
                49,
                59,
                15,
                15,
                14,
                15,
                8,
                7,
                8,
            ]
        ]
    )
    model(sample_input).argmax(dim=-1)
    print(sample_output)

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
    temp_board_state = temp_board_state.detach().numpy()

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

    focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].cpu())

    full_linear_probe = torch.load(
        OTHELLO_ROOT / "main_linear_probe.pth", map_location="cpu"
    )
    rows = 8

    cols = 8
    options = 3
    black_to_play_index = 0
    white_to_play_index = 1
    blank_index = 0
    their_index = 1
    my_index = 2
    linear_probe = torch.zeros(cfg.d_model, rows, cols, options, device="cpu")
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
        torch.tensor(flipped_focus_states)
    )

    # Take the argmax
    focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)
    probe_out = einops.einsum(
        focus_cache["resid_post", 6],
        linear_probe,
        "game move d_model, d_model row col options -> game move row col options",
    )
    probe_out_value = probe_out.argmax(dim=-1)

    correct_middle_odd_answers = (
        probe_out_value.cpu() == focus_states_flipped_value[:, :-1]
    )[:, 5:-5:2]
    accuracies_odd = einops.reduce(
        correct_middle_odd_answers.float(), "game move row col -> row col", "mean"
    )
    correct_middle_answers = (
        probe_out_value.cpu() == focus_states_flipped_value[:, :-1]
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
    state = torch.zeros((64,), dtype=torch.float32, device="cpu") - 10.0
    state[stoi_indices] = focus_logits[game_index, pos].log_softmax(dim=-1)[1:]
    print("state:", state.shape)


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
