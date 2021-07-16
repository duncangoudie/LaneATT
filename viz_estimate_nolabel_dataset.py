"""
Given an unlabelled dataset (such as KITTI), visualise the result of images being run through the LaneATT model.

"""

import argparse

import cv2
import torch
import random
import numpy as np
from tqdm import tqdm, trange

from lib.config import Config

import os
import sys

PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from lib.datasets.lane_dataset import LaneDataset
from lib.datasets.nolabel_dataset import NoLabelDataset


class Visualiser():

    def __init__(self, model_path: str, image_folder: str):
        """

        @param model_path: eg "/path/to/models/model_0015.pt"
        """

        # config file and it's details
        self._cfg = Config("cfgs/eval/laneatt_culane_resnet18_evaluation.yml")
        self._test_parameters = self._cfg.get_test_parameters()

        self._device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

        self._model_path = model_path
        self._model = self._cfg.get_model()
        self._model.load_state_dict(torch.load(self._model_path)['model'])
        self._model = self._model.to(self._device)

        #img_w = 1226
        #img_h = 370
        #img_w = 1640
        #img_h = 590
        img_w = 640
        img_h = 360
        self._dataset_nolabel = LaneDataset(dataset='nolabel_dataset', img_size=(img_h,img_w), root=image_folder)
        #self._dataset_nolabel = NoLabelDataset(root=image_folder)
        self._test_dataloader = torch.utils.data.DataLoader(dataset=self._dataset_nolabel,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,)

    def visualise_results(self):
        """

        @return:
        """

        predictions = []
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(self._test_dataloader)):
                images = images.to(self._device)
                output = self._model(images, **self._test_parameters)
                prediction = self._model.decode(output, as_lanes=True)
                print("PREDICTION: ", prediction)
                predictions.extend(prediction)
                img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img, fp, fn = self._test_dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                cv2.imshow('pred', img)
                cv2.waitKey(0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        help="filepath to folder of images",
                        action="store")
    parser.add_argument("-m",
                        help="filepath to model",
                        action="store")

    args =parser.parse_args()
    if args.i is not None:
        image = args.i
        model_path = args.m

        v = Visualiser(model_path=model_path, image_folder=image)
        v.visualise_results()




    else:
        print('Incorrect arguments...')



if __name__== "__main__":
    main()