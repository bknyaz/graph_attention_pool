dataset="colors-3"
seed=111
checkpoints_dir=./checkpoints/
params="-D $dataset --test_batch_size 100 -K 2 -f 64,64 --aggregation mean --n_hidden 0 --readout sum --dropout 0 --threads 0 --pool_arch fc_prev --seed $seed --results $checkpoints_dir -d ./data"

logs_dir=./logs/

# Global pooling
python main.py $params --eval_attn_train --eval_attn_test --epochs 100 --lr_decay_step 90 | tee $logs_dir/"$dataset"_global_max_seed"$seed".log;

# Unsupervised attention
thresh=0.03
pool=unsup
python main.py $params --pool attn_"$pool"_threshold_skip_"$thresh" --epochs 300 --lr_decay_step 280 | tee $logs_dir/"$dataset"_"$pool"_seed"$seed".log;

# Supervised attention
thresh=0.05
pool=sup
python main.py $params --pool attn_"$pool"_threshold_skip_"$thresh" --epochs 300 --lr_decay_step 280 | tee $logs_dir/"$dataset"_"$pool"_seed"$seed".log;

# Weakly-supervised attention
thresh=0.05
python main.py $params --pool attn_sup_threshold_skip_"$thresh" --epochs 300 --lr_decay_step 280 --alpha_ws $checkpoints_dir/"$dataset"_alpha_WS_train_seed"$seed"_orig.pkl | tee $logs_dir/"$dataset"_weaksup_seed"$seed".log;
