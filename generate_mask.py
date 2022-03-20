#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import random
import argparse
from natsort import natsorted
import pdb
st = pdb.set_trace

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    # vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    vis_im = vis_parsing_anno_color

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im

@torch.no_grad()
def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth', part=0):
    random.seed(0)
    num_frames = 20
    frame_step = 4

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join(cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    with open(f"list_videos_4-{part}.txt", 'r') as f:
        videos = [s.strip() for s in f.readlines()]

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    for folder_name in tqdm(videos):
        if folder_name.endswith('.txt'):
            continue
        folder_path = os.path.join(dspth, folder_name)
        frames = natsorted(os.listdir(folder_path))
        # frames = random.sample(frames, num_frames)
        frames = frames[::frame_step]
        os.makedirs(osp.join(respth, folder_name), exist_ok=True)
        for frame in frames:
            image_path = os.path.join(folder_path, frame)
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # print(np.unique(parsing))
            mask = vis_parsing_maps(image, parsing, stride=1, save_im=False, save_path=None)
            cv2.imwrite(osp.join(respth, folder_name, frame), mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, default=0)
    args = parser.parse_args()

    """
    CUDA_VISIBLE_DEVICES=0 python3 make_mask_vox_all.py --part 0
    """

    evaluate(dspth='MMVID/data/mmvoxceleb/video', respth='MMVID/data/mmvoxceleb/mask', cp='79999_iter.pth', part=args.part)
