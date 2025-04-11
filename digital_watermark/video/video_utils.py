import json

import cv2
from reedsolo import RSCodec, ReedSolomonError
import numpy as np


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    return frames


def is_low_motion(prev_frame, frame, threshold):

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)

    # Threshold the difference to get a binary image, 1 represents low motion, 0 represents motion
    low_motion_mask = np.where(diff < threshold, 1, 0).astype(np.uint8)

    return low_motion_mask
