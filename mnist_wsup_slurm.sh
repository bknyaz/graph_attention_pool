#!/bin/bash
#
#SBATCH -p gpu                # queue
#SBATCH --ntasks 1            # number of tasks
#SBATCH --mem 16000           # memory pool per process
#SBATCH -o results/mnist/slurm/slurm.%N.%j.mnist.out    # STDOUT
#SBATCH -t 72:00:00            # time (D-HH:MM)
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 16000

source activate py3-torch1.0-cuda10
which python
nvidia-smi
results_dir=/export/mlrg/bknyazev/projects/graph_attention_pool/results/mnist/
seed=$(( ( RANDOM % 10000 )  + 1 ))

echo $seed

params="--epochs 30 --lr_decay_step 20,25 --test_batch_size 50 -K 10 --aggregation mean -D mnist --n_hidden 0 --readout max -f 4,64,512 --dropout 0.5 --threads 0 --img_features mean,coord --img_noise_levels 0.5,0.75 --pool_arch fc_prev --kl_weight 100"

thresh=0.001
python main.py  $params --seed $seed --pool attn_gt_threshold_skip_skip_0 --results $results_dir/gt;
for pool in sup unsup;
  do python main.py  $params --seed $seed --pool attn_"$pool"_threshold_skip_skip_"$thresh" --results $results_dir/"$pool";
done
python main.py  $params --seed $seed --eval_attn_train --eval_attn_test --results $results_dir/global_max;
python main.py  $params --seed $seed --pool attn_sup_threshold_skip_skip_"$thresh" --alpha_ws $results_dir/global_max/alpha_WS_train_seed"$seed"_orig.pkl --results $results_dir/wsup;
