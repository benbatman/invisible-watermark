import unittest
import os
import tempfile
from unittest.mock import patch


import cv2
import numpy as np
from reedsolo import RSCodec

from digital_watermark.video.video_watermark import VideoWatermark
from digital_watermark.shared_utils import (
    select_random_blocks,
    create_coeff_pairs,
    embed_with_pairs,
)

from digital_watermark.constants import JPEG_QUANT_TABLE


class TestVideoWatermark(unittest.TestCase):
    def setUp(self):
        pass
