import unittest
import os
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
from reedsolo import RSCodec

from digital_watermark.image.image_watermark import ImageWatermark


class TestImageWatermark(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test images
        self.test_dir = tempfile.mkdtemp()

        # Create a test image
        self.test_image_path = os.path.join(self.test_dir, "test_image.jpg")
        test_image = np.ones((64, 64, 3), dtype=np.uint8) * 255  # White image
        cv2.imwrite(self.test_image_path, test_image)
