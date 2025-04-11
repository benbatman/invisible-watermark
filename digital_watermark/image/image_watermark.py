import hashlib
import logging

import numpy as np
import cv2
from reedsolo import RSCodec, ReedSolomonError

from digital_watermark.shared_utils import select_random_blocks
from digital_watermark.constants import JPEG_QUANT_TABLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageWatermark:
    def __init__(self, watermark: str = "", ecc_symbols=20):
        """
        Initialize the ImageWatermark class.
        If no watermark (cp2a data, etc.) is provided, a hash will be generated from the image data.
        Parameters:
        -----------
        watermark: str
            The watermark string to be embedded in the image.
        ecc_symbols: int
            The number of error correction symbols to use for the Reed-Solomon encoding.
        """
        self.watermark = watermark
        self.ecc_symbols = ecc_symbols
        self.quant_table = JPEG_QUANT_TABLE

    def generate_rc_encoded_hash_from_image(self, image_data: cv2.typing.MatLike):
        """
        Takes in image data and returns a sha256 hash object.
        This is an alterantive to a user supplied watermark.

        Returns:
        ---------
        encoded_hash: bytearray(b'')
            The RC encoded hash
        bit_hash: str
            The binary representation of the encoded hash
        hash_length: int
            The length of the bit hash
        """

        image_bytes = image_data.tobytes()
        hash_object = hashlib.sha256(image_bytes)

        rs = RSCodec(self.ecc_symbols)
        binary_digest = hash_object.digest()  # Binary digest
        print("Binary digest", binary_digest)
        print("Length of binary digest: ", len(binary_digest))
        encoded_hash = rs.encode(binary_digest)
        print("RC Encoded hash", encoded_hash)
        print("Length of RC encoded hash: ", len(encoded_hash))
        bit_hash = "".join(format(byte, "08b") for byte in encoded_hash)
        print("Bit hash", bit_hash)
        print("Length of bit hash: ", len(bit_hash))
        hash_length = len(bit_hash)

        return encoded_hash, bit_hash, hash_length

    def _create_coeff_pairs(self, channel, hash_length, seed=42):
        np.random.seed(seed)
        # print("Hash length: ", hash_length)
        total_coeffs = channel.size
        coeff_pairs = []
        sorted_indices = np.argsort(self.quant_table.flatten())
        # Get lower frequency indices by indexed closer to the beginning of the array
        low_freq_indices = sorted_indices[: int(total_coeffs * 0.4)]

        if hash_length > len(low_freq_indices):
            raise ValueError(
                "Not enough coefficients to create required number of pairs"
            )

        # select unique indices upfront from the low frequency indices
        selected_indices = np.random.choice(
            low_freq_indices, hash_length * 2, replace=False
        )

        # Pair up the selected indices (reshape the flat array into a 2-column array for pairs)
        coeff_pairs = selected_indices.reshape(-1, 2)

        return coeff_pairs.tolist()  # convert back to list of tuples

    def _embed_with_pairs(self, cH, binary_hash, coeff_pairs, seed=42):
        np.random.seed(seed)
        pos_range = (10.0, 20.5)  # Positive differential range for '1'
        neg_range = (-20.5, -9.9)  # Negative differential range for '0'

        for bit, (idx1, idx2) in zip(binary_hash, coeff_pairs):
            current_diff = cH.flat[idx1] - cH.flat[idx2]
            # print("Current differential: ", current_diff)
            if bit == "1":
                target_diff = round(np.random.uniform(*pos_range), 2)
            else:
                # Negative range for 0
                target_diff = round(np.random.uniform(*neg_range), 2)

            # Adjust the first coefficient to achieve the desired differential
            # print(f"cH.flat[idx1] {cH.flat[idx1]} | cH.flat[idx2] {cH.flat[idx2]} Target diff {target_diff}")
            cH.flat[idx1] = np.round(cH.flat[idx2] + target_diff, 2)
            post_diff = cH.flat[idx1] - cH.flat[idx2]
            # print(f"Post diff: {post_diff}")
            if post_diff > 20.5:
                print("\n\nPost diff too large", post_diff)
            elif post_diff < -20.5:
                print("\n\nPost diff too small", post_diff)
            # Make sure the diff is in the range we want it to be for a given 1 or 0
            assert abs(post_diff - target_diff) < 0.01, "Diffs not equal"

        return cH

    def _extract_segment(self, channel, coeff_pairs, seed=42):
        """
        Extracts the segment from the channel using the coefficient pairs.
        """
        np.random.seed(seed)

        binary_hash = ""
        for idx1, idx2 in coeff_pairs:
            # print("dct 1 value: ", channel.flat[idx1])
            # print("dct 2 value: ", channel.flat[idx2])
            diff = round(channel.flat[idx1] - channel.flat[idx2], 2)
            # print("Diff: ", diff)
            # Adjust diff value to be in the range we want it to be in case jpeg compression caused errors

            # print("Diff after adjustment: ", diff)
            if diff > 0:
                binary_hash += "1"
            elif diff < 0:
                binary_hash += "0"

        return binary_hash

    def embed_watermark(
        self, image_upload_path: str, image_download_path: str = "", ecc_symbols=20
    ):
        """
        Embed the hash into the image using the Y channel of the YCrCb color space.

        Parameters:
        -----------
        image_upload_path: str
            The path to the image to be watermarked.
        image_download_path: str
            The path to save the watermarked image.
        ecc_symbols: int
            The number of error correction symbols to use for the Reed-Solomon encoding.
        """
        image = cv2.imread(image_upload_path)
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        encoded_hash, bit_hash, hash_length = self.generate_rc_encoded_hash_from_image(
            image
        )
        y_channel = ycrcb_image[:, :, 0]
        NUM_BLOCKS = (y_channel.shape[0] // 8) * (y_channel.shape[1] // 8)
        NUM_SELECTED_BLOCKS = len(bit_hash)
        selected_blocks = select_random_blocks(NUM_BLOCKS, NUM_SELECTED_BLOCKS)
        segment_length = min(len(bit_hash) // len(selected_blocks), 8)
        # Process the Y channel in 8x8 blocks
        h, w = y_channel.shape
        hash_index = 0
        for row in range(0, h, 8):
            for col in range(0, w, 8):
                if hash_index >= len(bit_hash):
                    break

                current_segment = bit_hash[hash_index : hash_index + segment_length]
                block = (
                    y_channel[row : row + 8, col : col + 8].astype(float) - 128
                )  # Zero shift
                dct_block = cv2.dct(block)
                # Create coefficient pairs from the low frequency parts of the image
                coeff_pairs = self._create_coeff_pairs(dct_block, len(current_segment))
                dct_block = self._embed_with_pairs(
                    dct_block, current_segment, coeff_pairs
                )

                # IDCT
                block = cv2.idct(dct_block) + 128  # Zero shift
                y_channel[row : row + 8, col : col + 8] = np.clip(block, 0, 255).astype(
                    np.uint8
                )

                hash_index += segment_length

        ycrcb_image[:, :, 0] = y_channel
        watermarked_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(image_download_path, watermarked_image)

        self._validate_hash(image_download_path, bit_hash, segment_length)

    def extract_hash(self, image_path: str, bit_hash: str, segment_length: int):
        watermarked_image = cv2.imread(image_path)
        ycrcb_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_image[:, :, 0]

        h, w = y_channel.shape
        extracted_hash = ""
        hash_index = 0
        finished = False
        for row in range(0, h, 8):
            if finished:
                break
            for col in range(0, w, 8):
                if hash_index >= len(bit_hash):
                    print("hash_index upon finishing: ", hash_index)
                    finished = True
                    break

                current_segment = bit_hash[hash_index : hash_index + segment_length]
                block = (
                    y_channel[row : row + 8, col : col + 8].astype(float) - 128
                )  # Zero shift
                dct_block = cv2.dct(block)
                coeff_pairs = self._create_coeff_pairs(dct_block, len(current_segment))
                extracted_segment = self._extract_segment(dct_block, coeff_pairs)
                extracted_hash += extracted_segment
                hash_index += segment_length
        return extracted_hash

    def _validate_hash(self, image_path: str, bit_hash: str, segment_length: int):
        """
        Validate the hash in the image.
        """
        extracted_hash = self.extract_hash(image_path, bit_hash, segment_length)

        print("Extracted hash: ", extracted_hash)
        print("Length of extracted hash: ", len(extracted_hash))
        hash_bytes = bytes(
            [
                int(extracted_hash[i : i + 8], 2)
                for i in range(0, len(extracted_hash), 8)
            ]
        )
        print("Length of hash bytes: ", len(hash_bytes))
        print("Hash bytes: ", hash_bytes)

        rs = RSCodec(self.ecc_symbols)
        try:
            decoded_hash, _, _ = rs.decode(hash_bytes)
            print("Length of decoded_hash: ", len(decoded_hash))
            if len(decoded_hash) == 32:
                return decoded_hash
        except ReedSolomonError as e:
            print("Error decoding hash:", e)

        return None
