#!/bin/bash

# To reconstruct the experiments in the paper download and use the complete imagenet 2012 dataset from https://www.image-net.org/. 
# Otherwise when using the provided subset please set --target-label to a number between 0 and 9 for imagenet experiments
# For devices other than CUDA please change cuda to cpu

timestamp=$(date +'%Y%m%d%H%M')

echo -e "\n\nPREPARE RUN IMDB\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/prepare_run.py --device cuda --data-class imdb | tee -a ./run_logs/${timestamp}_run.txt;

echo VANILLA IMDB | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imdb --loss-type vanilla --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nVIB IMDB\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imdb --loss-type vib --num-runs 5 --betas 0.0001 0.0005 0.001 0.005  0.01 0.05 0.1 0.5 --num-epochs 150 --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nVCEB IMDB\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imdb --loss-type ceb --num-runs 5 --betas 1 2 3 4 5 --num-epochs 150 --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nSVIB IMDB\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imdb --loss-type vub --num-runs 5 --betas 0.0001 0.001 0.01 0.1 --num-epochs 150 --device cuda --lambdas 1 2 3 | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nPREPARE RUN IMAGENET\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/prepare_run.py --device cuda --data-class imagenet | tee -a ./run_logs/${timestamp}_run.txt;

echo VANILLA IMAGENET | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imagenet --loss-type vanilla --target-label 805 --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nVCEB IMAGENET\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imagenet --loss-type ceb --num-runs 5 --betas 1 2 3 4 5 --num-epochs 100 --target-label 805 --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nVIB IMAGENET\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imagenet --loss-type vib --num-runs 5 --betas 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 --num-epochs 100 --target-label 805 --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nSVIB IMAGENET\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imagenet --loss-type vub --num-runs 5 --betas 0.0001 0.001 0.01 0.1 --num-epochs 100 --target-label 805 --device cuda --lambdas 1 2 3 | tee -a ./run_logs/${timestamp}_run.txt;
