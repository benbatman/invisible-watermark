import numpy as np
import cv2
import pywt
import hashlib
import pickle
import os
import random

import matplotlib.pyplot as plt
from scipy.linalg import svd
from skimage.util import view_as_blocks
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from datetime import datetime
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class ImageZeroWatermark:
    def __init__(self):
        pass
