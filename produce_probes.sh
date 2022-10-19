for X in {0..8}
do

CUDA_VISIBLE_DEVICES=0 python train_probe_othello.py --layer $X --random
CUDA_VISIBLE_DEVICES=0 python train_probe_othello.py --layer $X --championship
CUDA_VISIBLE_DEVICES=0 python train_probe_othello.py --layer $X

for Y in {2,4,8,16,32,64,128,256,512}
do
CUDA_VISIBLE_DEVICES=0 python train_probe_othello.py --layer $X --twolayer --mid_dim $layer --random
CUDA_VISIBLE_DEVICES=0 python train_probe_othello.py --layer $X --twolayer --mid_dim $layer --championship
CUDA_VISIBLE_DEVICES=0 python train_probe_othello.py --layer $X --twolayer --mid_dim $layer
done

done