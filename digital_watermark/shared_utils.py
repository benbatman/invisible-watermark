import numpy as np


def select_random_blocks(num_blocks, num_to_select, seed=42):
    np.random.seed(seed)
    selected_indices = np.random.choice(num_blocks, num_to_select, replace=False)
    return selected_indices


def create_coeff_pairs(self, channel, hash_length, seed=42):
    np.random.seed(seed)
    # print("Hash length: ", hash_length)
    total_coeffs = channel.size
    coeff_pairs = []
    sorted_indices = np.argsort(self.quant_table.flatten())
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
