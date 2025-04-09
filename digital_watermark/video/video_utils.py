import json

import cv2
from reedsolo import RSCodec, ReedSolomonError
import numpy as np


def binarize_and_encode_watermark(watermark: str, ecc_symbols: int = 20):
    """
    Takes in a JSON string and returns the binary representation of the encoded hash.
    Parameters:
    -----------
    json_str: str
        The JSON string to be encoded.
    ecc_symbols: int
        The number of error correction symbols to use for the Reed-Solomon encoding.
    Returns:
    ---------
    bit_str: str
        The binary representation of the encoded hash.
    byte_str: bytes
        The byte representation of the encoded hash.
    bit_hash_length: int
        The length of the bit hash.
    """
    rs = RSCodec(ecc_symbols)
    byte_str = watermark.encode("utf-8")
    encoded_bytes = rs.encode(byte_str)

    bit_str = "".join(format(byte, "08b") for byte in encoded_bytes)
    print(bit_str)
    print(len(bit_str))

    bytes_list = [int(bit_str[i : i + 8], 2) for i in range(0, len(bit_str), 8)]
    byte_str = bytes(bytes_list)
    print(byte_str)
    print(len(byte_str))
    bit_hash_length = len(bit_str)
    return bit_str, byte_str, bit_hash_length


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
