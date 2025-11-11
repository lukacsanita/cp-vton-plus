import subprocess
import shutil
import time
import argparse
import os
import cv2

import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow

if __name__ == "__main__":
    # Parse user options
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", help="Name of the running mode (train, test or demo)")
    parser.add_argument("--data_list", default="test_pairs.txt")
    opt = parser.parse_args()
    print(opt)
    
    # Record start time
    t = time.time()
    
    # Define commands for GMM training or testing
    if opt.mode in ["test", "demo"]:
        gmm_run = (f"python test.py --name GMM --stage GMM --workers 1 --b 2 --datamode test --data_list {opt.data_list} "
                    "--checkpoint checkpoints/GMM/gmm_final.pth")
    elif opt.mode == "train":
        gmm_run = "python train.py --name GMM --stage GMM --workers 4 --save_count 5000 --shuffle"    

    # Run GMM
    subprocess.call(gmm_run, shell=True)

    # Define source and destination paths for warp data
    # A GMM kimenetéből származó warp-cloth és warp-mask mappákat bemásolja a TOM-hoz, mert az ezekből generálja a végső képet.
    warp_cloth_src = "result/GMM/test/warp-cloth"
    warp_mask_src = "result/GMM/test/warp-mask"
    warp_cloth_dst = f"data/{opt.mode}/warp-cloth"
    warp_mask_dst = f"data/{opt.mode}/warp-mask"

    # Copy warp data directories
    shutil.copytree(warp_cloth_src, warp_cloth_dst, dirs_exist_ok=True)
    shutil.copytree(warp_mask_src, warp_mask_dst, dirs_exist_ok=True)

    # Define commands for TOM training or testing
    if opt.mode in ["test", "demo"]:
        tom_run = (f"python test.py --name TOM --stage TOM --workers 1 --b 2 --datamode test --data_list {opt.data_list} "
                    "--checkpoint checkpoints/TOM/tom_final.pth")
    elif opt.mode == "train":
        tom_run = "python train.py --name TOM --stage TOM --workers 4 --save_count 5000 --shuffle"

    # Run TOM
    subprocess.call(tom_run, shell=True)
    
    # Print total execution time
    total_time = time.time() - t
    print(f"TOTAL TIME: {total_time} seconds")
    print("ALL MODELS PROCESS FINISHED")
    # Show result image when demo
    # Ha --mode demo, akkor megnyitja a result/TOM/test/try-on/ mappából az elkészült képet, és megmutatja egy OpenCV ablakban.
    # Ez helyi gépen működik, de Google Colabban nem fog megjelenni az imshow() ablak
    if opt.mode == "demo":
        demo_pair_path = f'data/{opt.data_list}'
        pair = open(demo_pair_path).readlines()[0]
        dst, src = pair.strip().split(' ')
        person = dst.split('_')[0]
        result_dir = "result/TOM/test/try-on"
        result_img_name = os.path.join(result_dir, person + "_0.jpg")

        # Read the image with opencv
        img = cv2.imread(result_img_name)
        # Show the image with window name
        cv2.imshow("Try-on", img) # colab: matplotlib.pyplot.imshow()
        # Window waits until user presses a key
        print("\n PRESS ANY KEY TO CLOSE THE WINDOW")
        cv2.waitKey(0)
        # Finally destroy/close all open windows
        cv2.destroyAllWindows()
