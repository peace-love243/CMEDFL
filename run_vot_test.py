import importlib
import os
import pickle
from pytracking.evaluation.environment import env_settings
import cv2
import numpy as np
import time
import torch
import argparse
import logging
import os.path as osp
import random

from pytracking.evaluation.otbdataset import OTBDataset
from votTester.vot import VOTTester


os.environ["CUDA_VISIBLE_DEVICES"] = "1" #################指定第一块gpu

def get_parameters(tracker_name, param_file_name):
    param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(tracker_name, param_file_name))
    params = param_module.parameters()
    return params

def read_image(image_file: str):
        return cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
#################################################################################
    torch.backends.cudnn.benchmark = False

    tracker_module_segm = importlib.import_module('pytracking.tracker.{}'.format('segm'))
    tracker_class_segm = tracker_module_segm.get_tracker_class()

    all_tests = [] # d3s  stm_train_tracker_segnum_nobackb_epoch10_vot2019 stm_train_tracker stm_train_tracker_segnum_noAuthorinit_epoch11_vot2018
    print('run vot test')
    for i in range(1):
        #pth_path = os.path.join('/home/jaffe/PycharmProjects/DMB/pytracking/networks', 'recurrent'+str(i+25)+'.pth.tar')
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format('segm', 'default_params'))
        params = param_module.parameters()
        segm_train_tracker = tracker_class_segm(params)

        testname = 'result_' + str(i)
        print(testname)
        all_tests.append(testname)
        testers1 = VOTTester(segm_train_tracker, testname, dataset_names = ['VOT2019'])
        testers1.test()

    testers1.pysot_eval.eval(all_tests)





