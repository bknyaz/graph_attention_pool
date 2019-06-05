params="--epochs 30 --lr_decay_step 20,25 --test_batch_size 100 -K 4 --aggregation mean -D mnist --n_hidden 0 --readout max -f 4,64,512 --dropout 0.5 --threads 0 --img_features mean,coord --img_noise_levels 0.5,0.75 --pool_arch fc_prev --kl_weight 100"

for pool in global_max unsup wsup sup gt;
  do mkdir results/$pool;
done

thresh=0.001

for i in $(seq 1 1 10);
  do seed=$(( ( RANDOM % 10000 )  + 1 ));
  python main.py  $params --seed $seed --eval_attn_train --eval_attn_test --results ./results/global_max | tee results/global_max/global_max_seed"$seed".log;
  python main.py  $params --seed $seed --pool attn_gt_threshold_skip_skip_0 --results ./results/gt | tee results/gt/gt_seed"$seed".log;
  for pool in unsup sup;
    do python main.py  $params --seed $seed --pool attn_"$pool"_threshold_skip_skip_"$thresh" --results ./results/"$pool" | tee results/"$pool"/"$pool"_seed"$seed".log;
  done
  python main.py  $params --seed $seed --pool attn_sup_threshold_skip_skip_"$thresh" --alpha_ws ./results/global_max/alpha_WS_train_seed"$seed".pkl --results ./results/wsup | tee results/wsup/wsup_seed"$seed".log;
done
