from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tqdm import tqdm
import numpy as np
from PIL import Image


# usage: mkdir_if_not_exist([root, dir])
def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def result(save_dir):
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
    
    width = 192
    height = 256
    nimgs = len(paths_dict.keys())
    for fn in tqdm(filelist):
        dst, src = fn.strip().split(' ')
        src = src.split('_')[0] # cloth
        dst = dst.split('_')[0] # model/image
        paddingimgs = Image.fromarray(np.zeros((height, width * nimgs,3)), mode='RGB')
        i = 0
        for path in paths_dict.keys():
            if i < 2:
                fn = src
            else:
                fn = dst
            imgname = os.path.join(path, fn + "_{}.jpg".format(paths_dict[path]))
            if 'image-parse' in path:
                imgname = os.path.join(path, fn + "_{}.png".format(paths_dict[path]))
            img = Image.open(imgname).convert('RGB')
            paddingimgs.paste(img, (width*i, 0))
            i += 1
        paddingimgs.save('{}/src_{}_dst_{}.png'.format(save_dir, src, dst))



if __name__ == '__main__':
    
    save_dir = 'smart_result'
    mkdir_if_not_exist([save_dir])
    result(save_dir)
