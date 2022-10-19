import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

from data.othello import permit, start_hands, OthelloBoardState, permit_reverse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

def print_board(labels):
    # torch tensor, [64], in 0--2
    bs_in_probe_mind = labels -1 
    anob = OthelloBoardState()
    anob.state = bs_in_probe_mind.detach().cpu().numpy().reshape(8, 8)
    anob.__print__()

def intervene(p, mid_act, labels_pre_intv, wtd, htd, plot=False):
    # p: probe model
    # mid_act: [512, ], the intervened, might not be at the lastest temporal position
    # labels_pre_intv, 
    # wtd: a dict of intervention_position, intervention_from, intervention_to
    # htd: a dict of some intervention parameters
    # plot: not supported yet
    # return a new_mid_act
    new_mid_act = torch.tensor(mid_act.detach().cpu().numpy()).cuda()
    new_mid_act.requires_grad = True
    opt = torch.optim.Adam([new_mid_act], lr=htd["lr"])

    labels_post_intv = labels_pre_intv.clone()
    weight_mask = htd["reg_strg"] * torch.ones(64).cuda()

    labels_post_intv[permit(wtd["intervention_position"])] = wtd["intervention_to"]
    weight_mask[permit(wtd["intervention_position"])] = 1

    logit_container = []
    loss_container = []
    for i in range(htd["steps"]): 
        opt.zero_grad()
        logits_running = p(new_mid_act[None, :])[0][0]  # [64, 3]
        logit_container.append(logits_running[permit(wtd["intervention_position"])].detach().cpu().numpy())
        loss = F.cross_entropy(logits_running, labels_post_intv, reduction="none")
        loss = torch.mean(weight_mask * loss)
        loss.backward()  # by torch semantics, loss is to be minimized
        loss_container.append(loss.item())
        opt.step()
    if 0:
        logits = np.stack(logit_container, axis=0)
        plt.plot(logits[:, 0], color="r", label="White")
        plt.plot(logits[:, 1], color="g", label="Blank")
        plt.plot(logits[:, 2], color="b", label="Black")
        plt.legend()
    labels_post_intv_hat = logits_running.detach().argmax(dim=-1)  # [64]
    num_error = torch.sum(labels_post_intv_hat - labels_post_intv).item()

    if plot:
        if num_error == 0:
            print(wtd["intervention_position"] + " Sucessfully intervened!")
        else:
            print(wtd["intervention_position"] + " Failed intervention! See the below two borads:")
            print("labels_post_intv_reality")
            print_board(labels_post_intv_hat)
            print("labels_post_intv_wished")
            print_board(labels_post_intv) 
    
    return new_mid_act
