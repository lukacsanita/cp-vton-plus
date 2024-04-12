import subprocess
import shutil
import time


# Record start time
t = time.time()

# Define commands for GMM training and testing
gmm_train = "python train.py --name GMM --stage GMM --workers 4 --save_count 5000 --shuffle"
gmm_test = ("python test.py --name GMM --stage GMM --workers 1 --b 2 --datamode test --data_list test_pairs.txt "
            "--checkpoint checkpoints/GMM/gmm_final.pth")

# Run GMM test
subprocess.call(gmm_test, shell=True)

# Define source and destination paths for warp data
warp_cloth_src = "result/GMM/test/warp-cloth"
warp_mask_src = "result/GMM/test/warp-mask"
warp_cloth_dst = "data/test/warp-cloth"
warp_mask_dst = "data/test/warp-mask"

# Copy warp data directories
shutil.copytree(warp_cloth_src, warp_cloth_dst, dirs_exist_ok=True)
shutil.copytree(warp_mask_src, warp_mask_dst, dirs_exist_ok=True)

# Define commands for TOM training and testing
tom_train = "python train.py --name TOM --stage TOM --workers 4 --save_count 5000 --shuffle"
tom_test = ("python test.py --name TOM --stage TOM --workers 1 --b 2 --datamode test --data_list test_pairs.txt "
            "--checkpoint checkpoints/TOM/tom_final.pth")

# Run TOM test
subprocess.call(tom_test, shell=True)

# Print total execution time
total_time = time.time() - t
print(f"TOTAL TIME: {total_time} seconds")
print("ALL PROCESS FINISHED")
