import unittest
import os
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
from reedsolo import RSCodec

from digital_watermark.image.image_watermark import ImageWatermark
from digital_watermark.shared_utils import (
    select_random_blocks,
    create_coeff_pairs,
    embed_with_pairs,
)
from digital_watermark.constants import JPEG_QUANT_TABLE


class TestImageWatermark(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # Create a test image
        self.test_image_path = os.path.join(self.test_dir, "test_image.jpg")
        test_image = np.ones((64, 64, 3), dtype=np.uint8) * 255  # White image
        cv2.imwrite(self.test_image_path, test_image)

        # Output path for watermarked images
        self.output_path = os.path.join(self.test_dir, "watermarked_image.jpg")

        # Create watermark instance
        self.watermark = ImageWatermark(ecc_symbols=10)

    def tearDown(self):
        # Clean up temp files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_hash_generation(self):
        """
        Test if the hash generation is correct.
        """
        image = cv2.imread(self.test_image_path)
        encoded_hash, bit_hash, hash_length = (
            self.watermark.generate_rc_encoded_hash_from_image(self.test_image_path)
        )

        # Check types
        self.assertIsInstance(encoded_hash, bytearray)
        self.assertIsInstance(bit_hash, str)
        self.assertIsInstance(hash_length, int)

        # Check lengths
        self.assertEqual(
            len(encoded_hash), 32 + self.watermark.ecc_symbols
        )  # 32 bytes for SHA256 + ecc symbols
        self.assertEqual(len(bit_hash), len(encoded_hash) * 8)  # 8 bits per byte
        self.assertEqual(hash_length, len(bit_hash))

        # Ensure bit hash is a binary string
        self.assertTrue(all(bit in "01" for bit in bit_hash))

    def test_coefficient_pairs_generation(self):
        """
        Test if the coefficient pairs generation is correct.
        """
        image = cv2.imread(self.test_image_path)
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_image[:, :, 0]
        block = y_channel[:8, :8].astype(float) - 128  # Zero shift
        dct_block = cv2.dct(block)

        # Test with different hash lengths
        for hash_length in [8, 16, 24]:
            coeff_pairs = create_coeff_pairs(dct_block, hash_length, JPEG_QUANT_TABLE)

            # Check that we have the correct number of pairs
            self.assertEqual(len(coeff_pairs), hash_length)

            # Check that all pairs are unique
            flat_indices = [idx for pair in coeff_pairs for idx in pair]
            self.assertEqual(len(flat_indices), len(set(flat_indices)))

    def test_embedding_and_extraction(self):
        """
        Test embedding and extracting watermark
        """
        self.watermark.embed_watermark(
            self.test_image_path, self.output_path, ecc_symbols=10
        )
        # Verify output file exists
        self.assertTrue(os.path.exists(self.output_path))
        # Load original and watermarked images
        original_img = cv2.imread(self.test_image_path)
        watermarked_img = cv2.imread(self.output_path)

        # Check dimensions
        self.assertEqual(original_img.shape, watermarked_img.shape)

        # Images should be slighly different but not too different
        diff = cv2.absdiff(original_img, watermarked_img)
        diff_sum = np.sum(diff)
        self.assertGreater(diff_sum, 0)  # Should be different
        self.assertLess(diff_sum, 20)  # Not too different

    @patch("hashlib.sha256")
    def test_deterministic_hash(self, mock_sha256):
        """
        Test if the hash generation is deterministic for same input
        """
        mock_digest = b"0" * 32
        mock_sha256.return_value.digest.return_value = mock_digest

        image = cv2.imread(self.test_image_path)
        encoded_hash1, bit_hash1, _ = (
            self.watermark.generate_rc_encoded_hash_from_image(image)
        )
        encoded_hash2, bit_hash2, _ = (
            self.watermark.generate_rc_encoded_hash_from_image(image)
        )

        # Same image should produce same hash
        self.assertEqual(encoded_hash1, encoded_hash2)
        self.assertEqual(bit_hash1, bit_hash2)

    def test_different_ecc_symbols(self):
        """
        Test if the hash generation is correct with different ecc symbols
        """
        for ecc_symbols in [10, 20, 30]:
            watermark = ImageWatermark(ecc_symbols=ecc_symbols)
            encoded_hash, bit_hash, _ = watermark.generate_rc_encoded_hash_from_image(
                self.test_image_path
            )
            self.assertEqual(len(encoded_hash), 32 + ecc_symbols)

    def test_edge_case_small_image(self):
        """
        Test if the watermarking works for small images
        """
        small_img_path = os.path.join(self.test_dir, "small_image.jpg")
        small_img = np.ones((16, 16, 3), dtype=np.uint8) * 255  # Small white image
        cv2.imwrite(small_img_path, small_img)

        # Need to test to see if it works on small images

    def test_custom_watermark(self):
        """
        Test if the custom watermarking works
        """
        custom_text = "Test Watermark"
        custom_watermark = ImageWatermark(watermark=custom_text)
        # Not finished yet


if __name__ == "__main__":
    unittest.main()
