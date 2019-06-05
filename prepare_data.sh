# Generate Colors data
for dim in 3 8 16 32; do python generate_data.py --dim $dim; done

# Generate Triangles data
python generate_data.py -D triangles --N_train 30000 --N_val 5000 --N_test 5000 --label_min 1 --label_max 10 --N_max 100 --threads 2

# Generate MNIST-75sp data
for split in train test; do python extract_superpixels.py -s $split -t 4; done
