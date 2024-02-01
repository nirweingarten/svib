#!/bin/bash

timestamp=$(date +'%Y%m%d%H%M')

echo -e "\n\nPREPARE RUN IMDB\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/prepare_run.py --device cuda --data-class imdb | tee -a ./run_logs/${timestamp}_run.txt;

echo VANILLA IMDB | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imdb --loss-type vanilla --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nVIB IMDB\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imdb --loss-type vib --num-runs 5 --betas 0.001 0.01 0.1 --num-epochs 150 --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nVUB IMDB\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imdb --loss-type vub --num-runs 5 --betas 0.001 0.01 0.1 --num-epochs 150 --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nPREPARE RUN IMAGENET\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/prepare_run.py --device cuda --data-class imagenet | tee -a ./run_logs/${timestamp}_run.txt;

echo VANILLA IMAGENET | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imdb --loss-type vanilla --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nVIB IMAGENET\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imagenet --loss-type vib --num-runs 5 --betas 0.001 0.01 0.1 --num-epochs 100 --device cuda | tee -a ./run_logs/${timestamp}_run.txt;

echo -e "\n\nVUB IMDB\n\n" | tee -a ./run_logs/${timestamp}_run.txt;

python src/train_and_eval_cdlvm.py --data-class imagenet --loss-type vub --num-runs 5 --betas 0.001 0.01 0.1 --num-epochs 100 --device cuda | tee -a ./run_logs/${timestamp}_run.txt;
