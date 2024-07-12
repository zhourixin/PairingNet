import subprocess

path = './'

run_file = path+"/PairingNet_train_val_test.py"
subprocess.run(["python", run_file, "--model_type=matching_train"])
subprocess.run(["python", run_file, "--model_type=matching_test"])
subprocess.run(["python", run_file, "--model_type=save_stage1_feature"])
cmd = 'python -m torch.distributed.launch --nproc_per_node 4 PairingNet_train_val_test.py --model_type=searching_train'
subprocess.run(cmd, shell=True, cwd=path)
cmd = 'python texture_countour_double_GCN.py --model_type=searching_test'
subprocess.run(cmd, shell=True, cwd=path)