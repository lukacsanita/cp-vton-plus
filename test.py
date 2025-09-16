# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_images, save_images

# Automatikusan lefuttatni az inference folyamatot egy adott tesztadatkészleten, és menteni a vizuális eredményeket képként (pl. warpolt ruha, próbálás eredménye stb.).

def get_opt():
    """
    Parses command line arguments for testing options.

    Returns:
        opt (argparse.Namespace): Namespace containing parsed arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="GMM", help="Name of the model (GMM or TOM)")

    parser.add_argument("--gpu_ids", default="", help="Comma separated list of GPU ids")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="test")

    parser.add_argument("--stage", default="GMM", help="Model stage (GMM or TOM)")

    parser.add_argument("--data_list", default="test_pairs.txt")
    # parser.add_argument("--data_list", default="test_pairs_same.txt")

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='Directory to save TensorBoard summaries')

    parser.add_argument('--result_dir', type=str,
                        default='result', help='Directory to save result images')

    parser.add_argument('--checkpoint', type=str, default='checkpoints/GMM/gmm_final.pth', help='Path to the model checkpoint')
    # parser.add_argument('--checkpoint', type=str, default='checkpoints/TOM/tom_final.pth', help='Path to the model checkpoint')

    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true', help='Shuffle input data')

    opt = parser.parse_args()
    return opt


def test_gmm(opt, test_loader, model, board):
    """
    Performs testing and visualization for the GMM model on a given dataset.

    Args:
        opt (argparse.Namespace): Parsed arguments.
        test_loader (CPDataLoader): Dataloader for the test set.
        model (nn.Module): GMM model.
        board (SummaryWriter): TensorBoard summary writer.
    """

    # Move model to GPU and set evaluation mode
    model.cuda()
    model.eval()

    # Create directories for different output images
    save_dir = os.path.join(opt.result_dir, opt.name, opt.datamode)
    os.makedirs(save_dir, exist_ok=True)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    os.makedirs(warp_cloth_dir, exist_ok=True)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    os.makedirs(warp_mask_dir, exist_ok=True)
    result_dir1 = os.path.join(save_dir, 'result_dir')
    os.makedirs(result_dir1, exist_ok=True)
    overlayed_TPS_dir = os.path.join(save_dir, 'overlayed_TPS')
    os.makedirs(overlayed_TPS_dir, exist_ok=True)
    warped_grid_dir = os.path.join(save_dir, 'warped_grid')
    os.makedirs(warped_grid_dir, exist_ok=True)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        shape_ori = inputs['shape_ori']  # original body shape without blurring

        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=True)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=True)
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=True)
        overlay = 0.7 * warped_cloth + 0.3 * im

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]

        # save_images(warped_cloth, c_names, warp_cloth_dir)
        # save_images(warped_mask*2-1, c_names, warp_mask_dir)
        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
        save_images(shape_ori.cuda() * 0.2 + warped_cloth *
                    0.8, im_names, result_dir1)
        save_images(warped_grid, im_names, warped_grid_dir)
        save_images(overlay, im_names, overlayed_TPS_dir)

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)


def test_tom(opt, test_loader, model, board):
    """
    Performs testing and visualization for the TOM model on a given dataset.

    Args:
        opt (object): Options object containing test parameters.
        test_loader (object): PyTorch dataloader for test data.
        model (nn.Module): TOM model.
        board (object): TensorBoard summary writer.
    """

    # Move model to GPU and set evaluation mode
    model.cuda()
    model.eval()

    # Create directories for different output images
    save_dir = os.path.join(opt.result_dir, opt.name, opt.datamode)
    os.makedirs(save_dir, exist_ok=True)
    try_on_dir = os.path.join(save_dir, 'try-on')
    os.makedirs(try_on_dir, exist_ok=True)
    p_rendered_dir = os.path.join(save_dir, 'p_rendered')
    os.makedirs(p_rendered_dir, exist_ok=True)
    m_composite_dir = os.path.join(save_dir, 'm_composite')
    os.makedirs(m_composite_dir, exist_ok=True)
    im_pose_dir = os.path.join(save_dir, 'im_pose')
    os.makedirs(im_pose_dir, exist_ok=True)
    shape_dir = os.path.join(save_dir, 'shape')
    os.makedirs(shape_dir, exist_ok=True)
    im_h_dir = os.path.join(save_dir, 'im_h')
    os.makedirs(im_h_dir, exist_ok=True)  # for test data

    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        # Extract data from dataloader
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        # Perform model inference
        # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
        outputs = model(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
        
        # Split outputs and apply activation functions
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        # Calculate final try-on image
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        # Define data for visualization
        visuals = [[im_h, shape, im_pose],
                   [c, 2*cm-1, m_composite],
                   [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names, try_on_dir)
        save_images(im_h, im_names, im_h_dir)
        save_images(shape, im_names, shape_dir)
        save_images(im_pose, im_names, im_pose_dir)
        save_images(m_composite, im_names, m_composite_dir)
        save_images(p_rendered, im_names, p_rendered_dir)  # for test data

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)



if __name__ == "__main__":
    # Parse user options
    opt = get_opt()
    print(opt)

    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))

    # Create dataset for testing
    test_dataset = CPDataset(opt)

    # Create dataloader
    test_loader = CPDataLoader(opt, test_dataset)

    # Setup TensorBoard for visualization
    os.makedirs(opt.tensorboard_dir, exist_ok=True)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))

    # Choose model and run test based on stage
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, test_loader, model, board)
    elif opt.stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, test_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished test %s, named: %s!' % (opt.stage, opt.name))
