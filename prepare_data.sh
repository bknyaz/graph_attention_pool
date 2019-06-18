# Generate Colors data
out_dir=./data
for dim in 3 8 16 32; do python generate_data.py --dim $dim -o $out_dir; done

# Generate Triangles data
python generate_data.py -D triangles --N_train 30000 --N_val 5000 --N_test 5000 --label_min 1 --label_max 10 --N_max 100 --threads 0 -o $out_dir

# Generate MNIST-75sp data
for split in train test; do python extract_superpixels.py -s $split -t 2 -o $out_dir; done

# Generate CIFAR-10-150sp data
#for split in train test; do python extract_superpixels.py -D cifar10 -c 10 -n 150 -s $split -t 4 -o $out_dir; done

# Download and unzip COLLAB, PROTEINS and D&D
for dataset in COLLAB PROTEINS DD;
do wget https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/"$dataset".zip -P $out_dir; unzip  "$out_dir"/"$dataset".zip -d $out_dir; done
