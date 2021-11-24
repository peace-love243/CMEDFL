import os
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_dilation
import cv2

import torch
from torch.nn import functional as NF
from torchvision.transforms import functional as TF


def calc_uncertainty1(score):

    # seg shape: bs, obj_n, h, w
    uncertainty = torch.var(score, dim=1)
    # print(uncertainty)
    # uncertainty1 = torch.max(score, dim=1)
    # # print(uncertainty1)
    # uncertainty2 = torch.min(score, dim=1)
    # uncertainty3 = uncertainty1 - uncertainty2
    # print(uncertainty3)
    # uncertainty = torch.mean(score_top5[:, 0:3]) / (torch.mean(score_top5[:, 150:201]) + 1e-8)  # bs, h, w

    uncertainty = torch.exp(uncertainty)  # bs, 1, h, w
    # print(uncertainty)

    return uncertainty

def calc_uncertainty(score):

    # seg shape: bs, obj_n, h, w
    score_top, _ = score.topk(k=2, dim=1)
    uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)  # bs, h, w
    uncertainty = torch.exp(1 - uncertainty).unsqueeze(1)  # bs, 1, h, w
    return uncertainty



