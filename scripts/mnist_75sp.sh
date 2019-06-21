dataset="mnist-75sp"
seed=3026
checkpoints_dir=./checkpoints/
params="-D $dataset --epochs 30 --lr_decay_step 20,25 --test_batch_size 100 -K 4 -f 4,64,512 --aggregation mean --n_hidden 0 --readout max --dropout 0.5 --threads 0 --img_features mean,coord --img_noise_levels 0.4,0.6 --pool_arch fc_prev --kl_weight 100 --seed $seed --results $checkpoints_dir -d ./data"

logs_dir=./logs/
thresh=0.01

# Global pooling
python main.py $params --eval_attn_train --eval_attn_test | tee $logs_dir/"$dataset"_global_max_seed"$seed".log;

# Unsupervised and supervised attention
for pool in unsup sup;
  do python main.py $params --pool attn_"$pool"_threshold_skip_skip_"$thresh" | tee $logs_dir/"$dataset"_"$pool"_seed"$seed".log;
done

# Weakly-supervised attention
python main.py $params --pool attn_sup_threshold_skip_skip_"$thresh" --alpha_ws $checkpoints_dir/"$dataset"_alpha_WS_train_seed"$seed"_orig.pkl | tee $logs_dir/"$dataset"_weaksup_seed"$seed".log;
