#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --cuda --lr=5e-4 --patience=1 --valid-niter=50 --max-train-iter=300 --batch-size=4096 --dropout=.3
elif [ "$1" = "test" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py decode model1.bin outputs/test_outputslinux6.txt --cuda
elif [ "$1" = "dev" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train  --lr=5e-4
elif [ "$1" = "test_local" ]; then
	python run.py decode model.bin 
elif [ "$1" = "tensorboard" ]; then
	tensorboard --logdir runs --bind_all
else
	echo "Invalid Option Selected"
fi