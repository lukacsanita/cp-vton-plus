import subprocess
import shutil
import time
import argparse



def get_opt():
    """
    Parses command line arguments.

    Returns:
        opt (argparse.Namespace): Namespace containing parsed arguments.
    """


    return opt


if __name__ == "__main__":
    # Parse user options
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", help="Name of the running mode (train or test)")
    opt = parser.parse_args()
    print(opt)
    
    # Record start time
    t = time.time()
    
    # Define commands for GMM training or testing
    if opt.mode == "test":
        gmm_run = ("python test.py --name GMM --stage GMM --workers 1 --b 2 --datamode test --data_list test_pairs.txt "
                    "--checkpoint checkpoints/GMM/gmm_final.pth")
    elif opt.mode == "train":
        gmm_run = "python train.py --name GMM --stage GMM --workers 4 --save_count 5000 --shuffle"

    # Run GMM
    subprocess.call(gmm_run, shell=True)

    # Define source and destination paths for warp data
    warp_cloth_src = "result/GMM/test/warp-cloth"
    warp_mask_src = "result/GMM/test/warp-mask"
    warp_cloth_dst = f"data/{opt.mode}/warp-cloth"
    warp_mask_dst = f"data/{opt.mode}/warp-mask"

    # Copy warp data directories
    shutil.copytree(warp_cloth_src, warp_cloth_dst, dirs_exist_ok=True)
    shutil.copytree(warp_mask_src, warp_mask_dst, dirs_exist_ok=True)

    # Define commands for TOM training or testing
    if opt.mode == "test":
        tom_run = ("python test.py --name TOM --stage TOM --workers 1 --b 2 --datamode test --data_list test_pairs.txt "
                    "--checkpoint checkpoints/TOM/tom_final.pth")
    elif opt.mode == "train":
        tom_run = "python train.py --name TOM --stage TOM --workers 4 --save_count 5000 --shuffle"

    # Run TOM
    subprocess.call(tom_run, shell=True)

    # Print total execution time
    total_time = time.time() - t
    print(f"TOTAL TIME: {total_time} seconds")
    print("ALL PROCESS FINISHED")
