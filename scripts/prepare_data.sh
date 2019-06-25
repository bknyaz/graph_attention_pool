date
seed=111

# Generate Colors data
out_dir=./data
for dim in 3 8 16 32; do python generate_data.py --dim $dim -o $out_dir --seed $seed; done

# Generate Triangles data
python generate_data.py -D triangles --N_train 30000 --N_val 5000 --N_test 5000 --label_min 1 --label_max 10 --N_max 100 -o $out_dir --seed $seed

# Generate MNIST-75sp data
for split in train test; do python extract_superpixels.py -s $split -o $out_dir --seed $seed; done

# Generate CIFAR-10-150sp data
#for split in train test; do python extract_superpixels.py -D cifar10 -c 10 -n 150 -s $split -t 0 -o $out_dir; done

# Generate noise for MNIST-75sp
python -c "import sys,torch; print('seed=%s\nout file noise=%s\nout file color noise=%s' % (sys.argv[1], sys.argv[2], sys.argv[3])); torch.manual_seed(int(sys.argv[1])); noise=torch.randn(10000,75, dtype=torch.float); torch.save(noise, sys.argv[2]); colornoise=torch.randn(10000,75,3, dtype=torch.float); torch.save(colornoise, sys.argv[3]);" $seed "$out_dir"/mnist_75sp_noise.pt "$out_dir"/mnist_75sp_color_noise.pt

# Download and unzip COLLAB, PROTEINS and D&D
for dataset in COLLAB PROTEINS DD;
do wget https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/"$dataset".zip -P $out_dir; unzip  "$out_dir"/"$dataset".zip -d $out_dir; done

date
