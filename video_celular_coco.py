import cv2
import numpy as np

from tools import iter_cel_phone_frames
from pycocotools.coco import COCO


ann_file = r'E:\desenvolvimento\coco_annotations\person_keypoints_val2017.json'

coco = COCO(ann_file)

for frame in iter_cel_phone_frames():
    ann_ids = coco.getAnnIds()
