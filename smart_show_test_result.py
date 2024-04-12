import os
from tqdm import tqdm
import numpy as np
from PIL import Image


def mkdir_if_not_exist(path):
    """
    This function checks if a directory path exists. 
    If it doesn't exist, it creates the directory structure.

    Args:
        path (list): A list of strings representing the directory path components.
    """
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def combine_results(save_dir):
    """
    This function takes a list of source and destination file paths as input. 
    It then processes them using CP-VTON+.
    The function generates result images and combines them into a single image for easier visualization.

    Args:
        save_dir (str): The directory path to save the combined image.
    """

    filename = 'data/test_pairs.txt'
    filelist = open(filename).readlines()
    
    paths_dict = {'data/test/cloth' : 1,
                  'data/test/cloth-mask' : 1,
                  'data/test/image' : 0,
                  'data/test/image-parse' : 0, # png
                  'result/GMM/test/warp-cloth' : 0,
                  'result/GMM/test/warp-mask' : 0,
                  'result/TOM/test/try-on' : 0,
                  }
    
    # Define image width and height
    width = 192
    height = 256

    # Get the number of image paths from the dictionary
    nimgs = len(paths_dict.keys())

    for fn in tqdm(filelist):
        # Extract the file names for destination person and source cloth
        # We need to put the cloth on the model
        dst, src = fn.strip().split(' ')
        src = src.split('_')[0] # cloth
        dst = dst.split('_')[0] # person
        
        # Create a new empty image for combining multiple images
        paddingimgs = Image.fromarray(np.zeros((height, width * nimgs,3)), mode='RGB')
        
        i = 0
        for path in paths_dict.keys():
            # Use source filename for the first two paths (cloth and cloth-mask)
            if i < 2:
                fname = src
            else:
                fname = dst
            imgname = os.path.join(path, fname + "_{}.jpg".format(paths_dict[path]))
            
            # Handle image-parse path as PNG format
            if 'image-parse' in path:
                imgname = os.path.join(path, fname + "_{}.png".format(paths_dict[path]))
            
            # Paste the image onto the combined image at specific position
            img = Image.open(imgname).convert('RGB')
            paddingimgs.paste(img, (width*i, 0))
            i += 1

        # Save the combined image with source and destination names
        paddingimgs.save('{}/src_{}_dst_{}.png'.format(save_dir, src, dst))



if __name__ == '__main__':
    """
    This block runs the script only if executed directly.
    """
    
    save_dir = 'smart_result'
    mkdir_if_not_exist([save_dir])
    combine_results(save_dir)
