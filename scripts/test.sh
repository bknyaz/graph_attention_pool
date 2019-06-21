TU_dir=/mnt/data/bknyazev/data/graph_data/
python main.py --seed 11 -D TU --cv_folds 2 --n_nodes 25 --epochs 3 --lr_decay_step 2 --test_batch_size 100 -f 16,17,18 -K 3 --aggregation mean --n_hidden 0 --readout max --dropout 0.1 --pool attn_sup_threshold_skip_skip_0 --pool_arch fc_prev -d $TU_dir/PROTEINS --debug
echo "Test 1 is passed!"

#python main.py -D triangles --epochs 100 --lr_decay_step 85,95 --test_batch_size 100 -f 64,64,64 -K 7 --aggregation sum --n_hidden 64 --readout max  --dropout 0 --pool attn_sup_threshold_skip_0.01_0.01 --pool_arch gnn_curr  --results None -d /scratch/ssd/data/graph_attention_pool/
