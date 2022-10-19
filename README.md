# Othello World

This repository provides the code for training, probing and intervening Othello-GPT.  
The implementation is based on [minGPT](https://github.com/karpathy/minGPT), thanks to Andrej Karpathy.

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

See `intervening_probe_interact_column.ipynb` for the intervention experiment, where we can customize (1) which model to intervene on, (2) the pre-intervention board state (3) which square(s) to intervene.

## Attribution via Intervention Plots

See `plot_attribution_via_intervention_othello.ipynb` for the attribution via intervention experiment, where we can also customize (1) which model to intervene on, (2) the pre-intervention board state (3) which square(s) to attribute.

## How to Cite

Coming soon
