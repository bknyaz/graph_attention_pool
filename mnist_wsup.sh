params="--epochs 30 --lr_decay_step 20,25 --test_batch_size 50 -K 10 --aggregation mean -D mnist --n_hidden 0 --readout max -f 4,64,512 --dropout 0.5 --threads 0 --img_features mean,coord --img_noise_levels 0.75,1.0 --pool_arch fc_prev --kl_weight 10"

results_dir=/mnt/data/bknyazev/checkpoints/
logs_dir=./results/mnist

for pool in global_max unsup wsup sup gt;
  do mkdir $logs_dir/$pool;
done

thresh=0.0001

for i in $(seq 1 1 10);
  do seed=$(( ( RANDOM % 10000 )  + 1 ));
  python main.py  $params --seed $seed --pool attn_gt_threshold_skip_skip_0 --results $results_dir/gt | tee $logs_dir/gt/gt_seed"$seed".log;
  for pool in sup unsup;
    do python main.py  $params --seed $seed --pool attn_"$pool"_threshold_skip_skip_"$thresh" --results $results_dir/"$pool" | tee $logs_dir/"$pool"/"$pool"_seed"$seed".log;
  done
  python main.py  $params --seed $seed --eval_attn_train --eval_attn_test --results $results_dir/global_max | tee $logs_dir/global_max/global_max_seed"$seed".log;
  python main.py  $params --seed $seed --pool attn_sup_threshold_skip_skip_"$thresh" --alpha_ws $results_dir/global_max/mnist_alpha_WS_train_seed"$seed"_orig.pkl --results $results_dir/wsup | tee $logs_dir/wsup/wsup_seed"$seed".log;
done
