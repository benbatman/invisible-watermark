import hashlib
from reedsolo import RSCodec, ReedSolomonError
import numpy as np


def select_random_blocks(num_blocks, num_to_select, seed=42):
    np.random.seed(seed)
    selected_indices = np.random.choice(num_blocks, num_to_select, replace=False)
    return selected_indices


def create_coeff_pairs_jpg(channel, hash_length, quant_table=JPEG_QUANT_TABLE, seed=42):
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
