import torch
from PIL import Image
import os


def prepare_image_for_board(img_tensor):
    """
    Prepares an image tensor for visualization on TensorBoard.
    This includes normalization and handling single-channel tensors.

    Args:
        img_tensor (torch.Tensor): The image tensor to prepare.

    Returns:
        torch.Tensor: The prepared image tensor.
    """
    # Normalizes an image tensor to the range [0, 1]
    tensor = (img_tensor.clone() + 1) * 0.5
    tensor.cpu().clamp(0, 1)

    # Handle single channel tensor
    if tensor.size(1) == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    return tensor


def create_image_grid(img_tensors_list):
    """
    Creates a grid of images from a list of image tensors.

    Args:
        img_tensors_list (list): A list containing image tensors.

    Returns:
        torch.Tensor: A tensor representing the combined image grid.
    """

    grid_h = len(img_tensors_list)
    grid_w = max(len(img_tensors) for img_tensors in img_tensors_list)

    batch_size, channel, height, width = prepare_image_for_board(img_tensors_list[0][0]).size()
    
    canvas_h = grid_h * height
    canvas_w = grid_w * width
    
    canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)

    for i, img_tensors in enumerate(img_tensors_list):
        for j, img_tensor in enumerate(img_tensors):
            offset_h = i * height
            offset_w = j * width
            tensor = prepare_image_for_board(img_tensor)
            canvas[:, :, offset_h: offset_h + height,
                   offset_w: offset_w + width].copy_(tensor)

    return canvas


def board_add_image(board, tag_name, img_tensor, step_count):
    """
    Adds a single image to the visualization board.

    Args:
        board: The TensorBoard.
        tag_name (str): The tag name for the image.
        img_tensor (torch.Tensor): The image tensor to add.
        step_count (int): The training/testing step count.
    """
    tensor = prepare_image_for_board(img_tensor)

    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)


def board_add_images(board, tag_name, img_tensors_list, step_count):
    """
    Adds a grid of images to the visualization board.

    Args:
        board: The TensorBoard.
        tag_name (str): The tag name for the image.
        img_tensors_list (list): The image tensors list to add.
        step_count (int): The training/testing step count.
    """

    tensor = create_image_grid(img_tensors_list)

    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)


def save_images(img_tensors, img_names, save_dir):
    """
    Saves a list of PyTorch tensors as images to a specified directory.

    Args:
        img_tensors (list): A list of PyTorch tensors representing images.
        img_names (list): A list of corresponding image names (strings).
        save_dir (str): The directory path to save the images.
    """

    for img_tensor, img_name in zip(img_tensors, img_names):
        # Normalize and convert tensor values to range [0, 255]
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        # Convert tensor to NumPy array and handle channel dimension
        array = tensor.numpy().astype('uint8')
        if array.shape[0] == 1:
            # Grayscale image - squeeze out unnecessary channel dimension
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            # RGB image - swap axes for correct channel order (CHW -> HWC)
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        # Save the image using Pillow
        Image.fromarray(array).save(os.path.join(save_dir, img_name))
