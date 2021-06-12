"""
    Compute depth maps for images in the input folder.
"""

import argparse
import imageio
import matplotlib.pyplot as plt
import os
import platform
import glob
import torch
import cv2
import numpy as np
from torchvision.transforms import Compose
from models.midas_net import MidasNet
from models.transforms import Resize, NormalizeImage, PrepareForNet

import sys

import matplotlib
matplotlib.use('Agg')

# RESIZE_W_1, RESIZE_H_1 = 640, 360
VIZ = False


def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def _minify(basedir, factors=[], resolutions=[]):
    '''
        Minify the images to small resolution for training
    '''

    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output
    import glob

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any(
        [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])

        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        print(ext)
        # sys.exit()
        img_path_list = glob.glob(os.path.join(imgdir, '*.%s' % ext))

        for img_path in img_path_list:
            save_path = img_path.replace('.jpg', '.png')
            img = cv2.imread(img_path)

            print(img.shape, r)
            # sys.exit()

            cv2.imwrite(save_path,
                        cv2.resize(img,
                                   (r[1], r[0]),
                                   interpolation=cv2.INTER_AREA))

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


def run(basedir, input_path, output_path, model_path,
        input_w=640, input_h=360, resize_height=288):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    img0 = [os.path.join(basedir, 'images', f)
            for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = cv2.imread(img0).shape
    height = resize_height
    factor = sh[0] / float(height)
    width = int(round(sh[1] / factor))
    _minify(basedir, resolutions=[[height, width]])

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)
    # sys.exit()

    small_img_dir = os.path.abspath(input_path + '_*x' + str(resize_height) + '/')
    print(glob.glob(small_img_dir)[0])

    small_img_path = sorted(
        glob.glob(glob.glob(small_img_dir)[0] + '/*.png'))[0]

    small_img = cv2.imread(small_img_path)

    # load network
    model = MidasNet(model_path, non_negative=True)

    transform_1 = Compose(
        [
            Resize(
                input_w,
                input_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_AREA,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.to(device)
    model.eval()

    # get input
    img_names = sorted(glob.glob(os.path.join(input_path, "*")))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind in range(len(img_names)):

        img_name = img_names[ind]
        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input
        img = read_image(img_name)
        img_input_1 = transform_1({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample_1 = torch.from_numpy(img_input_1).to(device).unsqueeze(0)
            prediction = model.forward(sample_1)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=[small_img.shape[0],
                          small_img.shape[1]],
                    mode="nearest",
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )

        if VIZ:
            if not os.path.exists('./midas_otuputs'):
                os.makedirs('./midas_otuputs')

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(prediction, cmap='jet')
            plt.savefig('./midas_otuputs/%s' % (img_name.split('/')[-1]))
            plt.close()

        print(filename + '.npy')
        np.save(filename + '.npy', prediction.astype(np.float32))

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        help='COLMAP Directory')
    parser.add_argument("--input_w", type=int, default=640,
                        help='input image width for monocular depth network')
    parser.add_argument("--input_h", type=int, default=360,
                        help='input image height for monocular depth network')
    parser.add_argument("--resize_height", type=int, default=288,
                        help='resized image height for training \
                        (width will be resized based on original aspect ratio)')

    args = parser.parse_args()
    BASE_DIR = args.data_path

    INPUT_PATH = BASE_DIR + "/images"
    OUTPUT_PATH = BASE_DIR + "/disp"

    MODEL_PATH = "model.pt"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(BASE_DIR, INPUT_PATH, OUTPUT_PATH, MODEL_PATH,
        args.input_w, args.input_h, args.resize_height)
