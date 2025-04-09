import hashlib
from argparse import ArgumentParser


import numpy as np
import cv2
from reedsolo import RSCodec, ReedSolomonError

from utils import select_random_blocks


class ImageWatermark:
    def __init__(self, watermark: str):
        self.watermark = watermark

    def generate_rc_encoded_hash_from_image(
        image_data: cv2.typing.MatLike, ecc_symbols=20
    ):
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

        rs = RSCodec(ecc_symbols)
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

    def embed_hash(
        self, image_upload_path: str, image_download_path: str, ecc_symbols=20
    ):
        image = cv2.imread(image_upload_path)
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        encoded_hash, bit_hash, hash_length = self.generate_rc_encoded_hash_from_image(
            image
        )
        y_channel = ycrcb_image[:, :, 0]
        NUM_BLOCKS = (y_channel.shape[0] // 8) * (y_channel.shape[1] // 8)
        NUM_SELECTED_BLOCKS = len(bit_hash)
        selected_blocks = select_random_blocks(NUM_BLOCKS, NUM_SELECTED_BLOCKS)

    def extract_hash(self):
        pass
