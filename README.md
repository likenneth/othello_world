### Update 02/13/2023 :fire::fire::fire:

Neel Nanda just released a [TransformerLens](https://github.com/neelnanda-io/TransformerLens) version of Othello-GPT ([Colab](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Othello_GPT.ipynb), [Repo Notebook](https://github.com/neelnanda-io/TransformerLens/blob/main/demos/Othello_GPT.ipynb)), boosting the mechanistic interpretability research of it. Based on his work, a tool was made to inspect each MLP neuron in Othello-GPT, e.g. see the differing activation for [neuron 255 in layer 3](https://kran.ai/othelloscope/L2/N255) and [neuron 250 in layer 8](https://kran.ai/othelloscope/L7/N250).

# Othello World

This repository provides the code for training, probing and intervening the Othello-GPT in [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382), to be present at ICLR 2023.  
The implementation is based on [minGPT](https://github.com/karpathy/minGPT), thanks to Andrej Karpathy.

## Abstract

> Language models show a surprising range of capabilities, but the source of their apparent competence is unclear. Do these networks just memorize a collection of surface statistics, or do they rely on internal representations of the process that generates the sequences they see? We investigate this question by applying a variant of the GPT model to the task of predicting legal moves in a simple board game, Othello. Although the network has no a priori knowledge of the game or its rules, we uncover evidence of an emergent nonlinear internal representation of the board state. Interventional experiments indicate this representation can be used to control the output of the network and create "latent saliency maps" that can help explain predictions in human terms.

## Table of Contents

1. [Installation](#installation)
2. [Training Othello-GPT](#training-othello-gpt)
3. [Probing Othello-GPT](#probing-othello-gpt)
4. [Intervening Othello-GPT](#intervening-othello-gpt)
5. [Attribution via Intervention Plots](#attribution-via-intervention-plots)
6. [How to Cite](#how-to-cite)

## Installation

Some plotting functions require Latex on your machine: check [this FAQ](https://github.com/garrettj403/SciencePlots/wiki/FAQ#installing-latex) for how to install.  
Then use these commands to set up: 
```
conda env create -f environment.yml
conda activate othello
python -m ipykernel install --user --name othello --display-name "othello"
mkdir -p ckpts/battery_othello
```

## Training Othello-GPT

Download the [championship dataset](https://drive.google.com/drive/folders/1KFtP7gfrjmaoCV-WFC4XrdVeOxy1KmXe?usp=sharing) and the [synthetic dataset](https://drive.google.com/drive/folders/1pDMdMrnxMRiDnUd-CNfRNvZCi7VXFRtv?usp=sharing) and save them in `data` subfolder.  
Then see `train_gpt_othello.ipynb` for the training and validation. Alternatively, checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1bpnwJnccpr9W-N_hzXSm59hT7Lij4HxZ?usp=sharing) to skip this step.  
The default experiment setting requires $8$ GPU's and takes up to roughly $12$ Gigabytes memory on each. Once you set up the code, we can use `jupyter nbconvert --execute --to notebook --allow-errors --ExecutePreprocessor.timeout=-1 train_gpt_othello.ipynb --inplace --output ckpts/checkpoint.ipynb` to run it in background.  

## Probing Othello-GPT

Then we will use `train_probe_othello.py` to train probes.  
For example, if we want to train a nonlinear probe with hidden size $64$ on internal representations extracted from layer $6$ of the Othello-GPT trained on the championship dataset, we can use the command `python train_probe_othello.py --layer 6 --twolayer --mid_dim 64 --championship`.  
Checkpoints will be saved to `ckpts/battery_othello` or can be alternatively downloaded from [here](https://drive.google.com/drive/folders/1uvj_M9ekHDJVdVOvMq828Z23AE7jZ01H?usp=sharing). What produces the these checkpoints are `produce_probes.sh`.  

## Intervening Othello-GPT

See `intervening_probe_interact_column.ipynb` for the intervention experiment, where we can customize (1) which model to intervene on, (2) the pre-intervention board state (3) which square(s) to intervene on.

## Attribution via Intervention Plots

See `plot_attribution_via_intervention_othello.ipynb` for the attribution via intervention experiment, where we can also customize (1) which model to intervene on, (2) the pre-intervention board state (3) which square(s) to attribute.

## How to Cite
```
@inproceedings{
li2023emergent,
title={Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task},
author={Kenneth Li and Aspen K Hopkins and David Bau and Fernanda Vi{\'e}gas and Hanspeter Pfister and Martin Wattenberg},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=DeG07_TcZvT}
}
```
