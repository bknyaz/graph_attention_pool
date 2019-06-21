dataset="triangles"
seed=111
checkpoints_dir=./checkpoints/
params="-D $dataset --epochs 100 --lr_decay_step 85,95 --test_batch_size 100 -K 7 -f 64,64,64 --aggregation sum --n_hidden 64 --readout max --dropout 0 --threads 0 --pool_arch gnn_curr --seed $seed --results $checkpoints_dir -d ./data"

logs_dir=./logs/

# Global pooling
python main.py $params --eval_attn_train --eval_attn_test | tee $logs_dir/"$dataset"_global_max_seed"$seed".log;

# Unsupervised attention
thresh=0.0001
pool=unsup
python main.py $params --pool attn_"$pool"_threshold_skip_"$thresh"_"$thresh" | tee $logs_dir/"$dataset"_"$pool"_seed"$seed".log;

# Supervised attention
thresh=0.001
pool=sup
python main.py $params --pool attn_"$pool"_threshold_skip_"$thresh"_"$thresh" | tee $logs_dir/"$dataset"_"$pool"_seed"$seed".log;

# Weakly-supervised attention
thresh=0.01
python main.py $params --pool attn_sup_threshold_skip_"$thresh"_"$thresh" --alpha_ws $checkpoints_dir/"$dataset"_alpha_WS_train_seed"$seed"_orig.pkl | tee $logs_dir/"$dataset"_weaksup_seed"$seed".log;
