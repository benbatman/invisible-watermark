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
    binarize_and_encode_watermark,
)


class TestVideoWatermark(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Create a test video
        self.test_video_path = os.path.join(self.test_dir, "test_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.test_video_path, fourcc, 20.0, (640, 480))

        for i in range(100):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            out.write(frame)
        out.release()

    def tearDown(self):
        # Clean up temp files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_hash_generation(self):
        """
        Test if the hash generation is correct.
        """
        watermark = "Test video watermark"
        encoded_hash, bit_hash, hash_length = binarize_and_encode_watermark(
            watermark, ecc_symbols=20
        )

        # Check types
        self.assertIsInstance(encoded_hash, bytearray)
        self.assertIsInstance(bit_hash, str)
        self.assertIsInstance(hash_length, int)

        # Check lengths
        self.assertEqual(len(encoded_hash), 32 + VideoWatermark.ecc_symbols)
        self.assertEqual(len(bit_hash), len(encoded_hash) * 8)
        self.assertEqual(hash_length, len(bit_hash))
        # Check if the hash is correctly generated
        self.assertEqual(
            bit_hash,
            "".join(format(byte, "08b") for byte in encoded_hash),
        )
        # Ensure bit hash is a binary string
        self.assertTrue(all(bit in "01" for bit in bit_hash))

    def test_embedding_and_extraction(self):
        """
        Test if the embedding and extraction process works correctly.
        """
        watermark = "Test video watermark"
        encoded_hash, bit_hash, hash_length = binarize_and_encode_watermark(
            watermark, ecc_symbols=20
        )

        # Create a VideoWatermark instance
        video_watermark = VideoWatermark(ecc_symbols=20)

        # Embed the watermark into the video
        watermarked_video_path = os.path.join(self.test_dir, "watermarked_video.mp4")
        video_watermark.embed_watermark(
            self.test_video_path,
            watermarked_video_path,
        )

        # Extract the watermark from the watermarked video
        extracted_hash = video_watermark.extract_watermark(watermarked_video_path)

        # Check if the extracted hash matches the original hash
        self.assertEqual(extracted_hash, bit_hash)

    def test_different_ecc_symbols(self):
        pass

    def test_invalid_video_format(self):
        pass

    def test_invalid_watermark(self):
        pass
