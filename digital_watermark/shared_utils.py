from typing import Tuple

import json

from reedsolo import RSCodec
import numpy as np


def select_random_blocks(num_blocks, num_to_select, seed=42):
    np.random.seed(seed)
    selected_indices = np.random.choice(num_blocks, num_to_select, replace=False)
    return selected_indices


def create_coeff_pairs(channel, hash_length, quant_table: np.ndarray, seed=42):
    np.random.seed(seed)
    # print("Hash length: ", hash_length)
    total_coeffs = channel.size
    coeff_pairs = []
    sorted_indices = np.argsort(quant_table.flatten())
    # Get lower frequency indices by indexed closer to the beginning of the array
    low_freq_indices = sorted_indices[: int(total_coeffs * 0.4)]

    if hash_length > len(low_freq_indices):
        raise ValueError("Not enough coefficients to create required number of pairs")

    # select unique indices upfront from the low frequency indices
    selected_indices = np.random.choice(
        low_freq_indices, hash_length * 2, replace=False
    )

    # Pair up the selected indices (reshape the flat array into a 2-column array for pairs)
    coeff_pairs = selected_indices.reshape(-1, 2)

    return coeff_pairs.tolist()  # convert back to list of tuples


def embed_with_pairs(self, cH, binary_hash, coeff_pairs, seed=42):
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


def extract_segment(self, channel, coeff_pairs, seed=42):
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


def binarize_and_encode_watermark(
    watermark: str, ecc_symbols: int = 20
) -> Tuple[str, bytes, int]:
    """
    Binarizes the watermark and encodes it using Reed-Solomon error correction.
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
