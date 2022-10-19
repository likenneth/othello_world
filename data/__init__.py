from .othello import get as get_othello
import seaborn as sns
import numpy as np
import torch

vv = .2

def plot_probs(ax, probs, valids):
    assert probs.numel() == 64
    probs = probs.detach().cpu().numpy().reshape(8, 8)
    annot = [f"{_:.2f}" for _ in probs.flatten().tolist()]
    for valid_index in valids:
        annot[valid_index] = ("\\underline{" + annot[valid_index] + "}")
#     print(annot)
    sns.heatmap(probs, ax=ax, vmin=0, vmax=vv, 
            yticklabels=list("ABCDEFGH"), xticklabels=list(range(1,9)), square=True, 
            annot=np.array(annot).reshape(8, 8), cmap=sns.color_palette("Blues", as_cmap=True), fmt="", cbar=False)
    return ax

def plot_mentals(ax, logits):
    assert logits.shape[0] == 64
    assert logits.shape[1] == 3
    probs = torch.softmax(logits, dim=-1)  # [64, 3]
    probs, preds = torch.max(probs, dim=-1)  # [64, ], [64, ]
    probs = probs.detach().cpu().numpy().reshape(8, 8)
    preds = preds.detach().cpu().numpy().reshape(8, 8)
    annot = []
    for ele in preds.flatten().tolist():
        if ele == 0:
            annot.append("O")
        elif ele == 1:
            annot.append(" ")
        else:
            annot.append("X")
    sns.heatmap(probs, ax=ax, vmin=0, vmax=1., 
            yticklabels=list("ABCDEFGH"), xticklabels=list(range(1,9)), square=True, 
            annot=np.array(annot).reshape(8, 8), cmap=sns.color_palette("Blues", as_cmap=True), fmt="", cbar=False)
    return ax