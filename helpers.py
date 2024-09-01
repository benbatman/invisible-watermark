import numpy as np
import cv2
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError
import hashlib
import pywt
from cryptography.fernet import Fernet
import sympy
import json


## NOTE:
# - JPEG compression primarily changes to the chrominance channels from the YCrCb so in theory, changes made to the Y channel should be affected less by jpg compression
# - downsampling is only applied to the chrominance channels, and the luminance channel keeps its original size.
# - Quantization is primarily applied to the high=frequency compoennts after DCT is applied
# - The general ratios bewteen the values are kept the same regardless of the compression level. The absolute values change drastically depending on compression level

# Standard quant table with compression of 50% for luminance component
JPEG_QUANT_TABLE = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)


def generate_token(hash: str):
    """
    Function to generate a key for encryption and decryption
    """
    key = Fernet.generate_key()
    f = Fernet(key)
    token = f.encrypt(hash.encode())
    return token


def extract_token_from_image(modified_image, token_length_bits):
    """
    Function to take in image and extract the embedded token from it.
    Currently doesn't work. :(
    """
    # Process has to be the same as used to encode the cryptographic hash in the original image
    # Convert the modified image to the YCrCb color space
    ycrcb_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2YCR_CB)

    # Extract the Y channel
    y_channel = ycrcb_image[:, :, 0].astype(float)

    # Apply the DCT to the Y channel
    h, w = y_channel.shape
    extracted_bits = ""
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = y_channel[i : i + 8, j : j + 8]
            dct_block = cv2.dct(block.astype(float))  # apply dct to block
            for x in range(8):
                for y in range(8):
                    if len(extracted_bits) < token_length_bits:
                        # Get the coefficient at the current position in the DCT block
                        coeff = dct_block[x, y]
                        # Determine the embedded bit based on the distance to the floor or ceiling value
                        # If the coefficient is closer to its ceiling value, assume a '1' was embedded
                        # Otherwise, assume a '0' was embedded.
                        extracted_bit = (
                            "1"
                            if (np.ceil(coeff) - coeff) <= (coeff - np.floor(coeff))
                            else "0"
                        )
                        # extracted_bit = '1' if coeff - np.floor(coeff) >= 0.5 else '0'
                        extracted_bits += extracted_bit
                    else:
                        break
                if len(extracted_bits) >= token_length_bits:
                    break
            if len(extracted_bits) >= token_length_bits:
                break

    # Convert the binary string to bytes
    # extracted_token_bytes = int(extracted_bits, 2).to_bytes((len(extracted_bits) + 7) // 8, byteorder='big')
    return extracted_bits


def lsb_embed(image, token):
    pass


#### DWT ####


def binarize_and_encode_json(json_file, ecc_symbols=20):

    with open(json_file) as f:
        data = json.load(f)

    rs = RSCodec(ecc_symbols)
    json_str = json.dumps(data)
    byte_str = json_str.encode("utf-8")
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


def correct_json(json_versions):
    corrected_json = {}
    keys = set(key for json_dict in json_versions for key in json_dict.keys())

    for key in keys:
        values = [json_dict[key] for json_dict in json_versions if key in json_dict]


def generate_rc_encoded_hash_from_image(image_data, ecc_symbols=20):
    """
    Takes in image data and returns a sha256 hash object

    Returns:
    ---------
    encoded_hash: bytearray(b'')
        The RC encoded hash
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


def adjust_to_nearest_prime(value):
    if value < 2:
        return 2  # The smallest prime
    prime = value
    while not sympy.isprime(prime):
        prime += 1
    return prime


def perform_dwt(image_data):
    # Check dtype of image data
    assert image_data.dtype == np.float64, "Image data must be of type np.float64"
    coeffs = pywt.dwt2(image_data, "haar")
    cA, (cH, cV, cD) = coeffs  # Decompose into approximation and details
    return cA, cH, cV, cD


def perform_idwt(cA, cH, cV, cD):
    coeffs = (cA, (cH, cV, cD))
    reconstructed_image = pywt.idwt2(coeffs, "haar")
    return reconstructed_image


def extract_hash_dwt(cH, hash_length=64):
    binary_hash = ""
    for i in range(hash_length * 8):  # Assuming SHA-256
        coeff = cH.flat[i]
        extracted_bit = (
            "1" if (np.ceil(coeff) - coeff) <= (coeff - np.floor(coeff)) else "0"
        )
        binary_hash += extracted_bit
        # if cH.flat[i] % 2 == 0:
        #     binary_hash += '0'
        # else:
        #     binary_hash += '1'

    # Return the original hash string from the binary representation.
    hash_string = "".join(
        chr(int(binary_hash[i : i + 8], 2)) for i in range(0, len(binary_hash), 8)
    )
    return hash_string


def extract_hash_dwt_rc(
    cH,
    binary_hash_length=416,
    threshold_factor=3,
    ecc_symbols=20,
    seed=42,
    delta1=0.9,
    delta0=0.5,
):
    np.random.seed(seed)
    # Choose indices in the mid-range of coefficient values for more robustness
    mid_range_indices = np.where(
        (cH.flat > np.percentile(cH, 25)) & (cH.flat < np.percentile(cH, 75))
    )[0]
    indices = np.random.choice(
        mid_range_indices, size=(256 + (ecc_symbols * 8)), replace=False
    )
    # print("indices: ", indices)

    # Select outlier indicies
    # indices = find_outlier_indices(cH.flat)[:binary_hash_length]
    print("Length of indices for decoding: ", len(indices))
    binary_hash = ""

    mad = np.median(np.abs(cH.flat - np.median(cH.flat)))
    threshold = threshold_factor * mad
    print("Threshold: ", threshold)

    delta_threshold = (delta1 + delta0) / 2
    median = np.median(cH.flat)

    # for idx in indices:
    #     coeff = cH.flat[idx]
    #     # extracted_bit = '1' if (np.ceil(coeff) - coeff) <= (coeff - np.floor(coeff)) else '0'

    #     # Consider Magnitude of the deviation from the original state
    #     # deviation = abs(coeff - np.round(coeff))
    #     # print("Deviation: ", deviation)

    #     # deviation_from_median = abs(coeff - median)
    #     # print("Deviation from median: ", deviation_from_median)
    #     fractional_part = abs(coeff - np.round(coeff))

    #     if fractional_part > delta1:
    #         extracted_bit = '1'
    #     else:
    #         extracted_bit = '0'
    #     binary_hash += extracted_bit

    for idx in indices:
        tenths_place = int((abs(cH.flat[idx]) * 10) % 10)
        rounded_value = np.round(cH.flat[idx])

        # print("Tenths place: ", tenths_place)

        # Determine the bit value baesd on the parity of the tenths place
        # extracted_bit = '1' if tenths_place % 2 == 0 else '0'
        # if rounded_value % 2 == 0:
        #     extracted_bit = '1'
        # else:
        #     extracted_bit = '0'

        print("Tenths place: ", tenths_place)
        if np.isclose(tenths_place, 2, atol=0.05):  # Using a tolerance for comparison
            extracted_bit = "1"
            print("Close enough, bit = 1")
        else:
            print("Not close enough, bit = 0")
            extracted_bit = "0"

        binary_hash += extracted_bit

    print("Extracted binary hash: ", binary_hash)
    print("Length of extracted binary hash: ", len(binary_hash))
    # Convert binary to bytes
    hash_bytes = bytes(
        [int(binary_hash[i : i + 8], 2) for i in range(0, len(binary_hash), 8)]
    )
    print("Bytes after conversion", hash_bytes)
    print("Length of bytes after conversion: ", len(hash_bytes))

    # Initialize RS codec and attempt to decode
    rs = RSCodec(ecc_symbols)
    try:
        decoded_hash = rs.decode(hash_bytes)
        print("Length of decoded_hash: ", len(decoded_hash))
        return decoded_hash
        # if len(decoded_hash) == 52:
        #     return decoded_hash
        # else:
        #     return None
    except ReedSolomonError as e:
        print("Error decoding hash:", e)
        return None


def find_outlier_indices(coefficients, threshold_factor=3):
    """Idenity indices of outlier coefficients based on MAD"""
    median = np.median(coefficients)
    mad = np.median(np.abs(coefficients - median))
    threshold = threshold_factor * mad
    outlier_indices = np.where(np.abs(coefficients - median) > threshold)[0]
    return outlier_indices


def embed_encoded_hash(
    cH, encoded_hash, threshold_factor=3, seed=42, delta1=0.9, delta0=0.5
):
    np.random.seed(seed)
    # Convert encoded hash to binary
    binary_hash = "".join(format(byte, "08b") for byte in encoded_hash)
    # Encoded hash is 42 bytes long coming into this function (32+10) 10 comes from the parity bytes from Reed Solomon

    # Choose indices in the mid-range of coefficient values for more robustness
    mid_range_indices = np.where(
        (cH.flat > np.percentile(cH, 25)) & (cH.flat < np.percentile(cH, 75))
    )[0]
    indices = np.random.choice(mid_range_indices, size=len(binary_hash), replace=False)
    # print("indices: ", indices)
    # # Get outlier indices
    # indices = find_outlier_indices(cH.flat, threshold_factor)

    # # Check to see if we have enough outliers to embed the entire binary hash
    # if len(indices) < len(binary_hash):
    #     raise ValueError("Not enough outliers to embed entire binary hash")

    # # Get subset of indices for the length of the binary hash
    # indices = indices[:len(binary_hash)]

    print("Length of indices for encoding: ", len(indices))
    print("Binary hash: ", binary_hash)
    print("Length of binary hash", len(binary_hash))
    print("max value of cH: ", max(cH.flat))
    print("min value of cH: ", min(cH.flat))

    # # Embed binary hash into the detail coeficients
    # for i, bit in enumerate(binary_hash):
    #     idx = indices[i]
    #     if bit == '1':
    #         # For '1', ensure an increase in magnitude
    #         cH.flat[idx] += delta1 if cH.flat[idx] >= 0 else -delta1
    #     else:
    #         # For '0', ensure a decrease in magnitude or less of an increase
    #         cH.flat[idx] -= delta0 if cH.flat[idx] > 0 else delta0

    # Try method of even-odd for 10s place
    for i, bit in enumerate(binary_hash):
        idx = indices[i]
        current = cH.flat[idx]
        # Get the whole part of the coefficient
        whole_part = np.floor(current) if current >= 0 else np.ceil(current)
        tenths_even = 0.2  # Choose an even tenths value
        tenths_odd = 0.1  # Choose an odd tenths value
        # Get tenths place
        tenths_place = int((abs(current) * 10) % 10)
        rounded_value = np.round(cH.flat[idx])
        # print("Tenths place: ", tenths_place)
        if bit == "1":
            # If odd, make it even
            adjustment = 0.1 if tenths_place % 2 != 0 else 0.0

            # Adjust based off of sign
            if whole_part >= 0.0:
                new_value = whole_part + tenths_even
            else:
                new_value = whole_part - tenths_even
            print("new value: ", new_value)
        else:
            # If even, make it odd
            # adjustment = (1 if tenths_place % 2 == 0 else 0) / 10.0
            adjustment = 0.1 if tenths_place % 2 == 0 else 0.0
            # Adjust based off of sign
            if whole_part >= 0.0:
                new_value = whole_part + tenths_odd
            else:
                new_value = whole_part - tenths_odd

            print("new value: ", new_value)

        # print("Adjustment: ", adjustment)
        # print("cH.flat[idx]: ", cH.flat[idx])
        # cH.flat[idx] += adjustment if cH.flat[idx] >= 0 else -adjustment
        # print("cH.flat[idx] after adjustment: ", cH.flat[idx])
        cH.flat[idx] = new_value
        print("cH.flat[idx] after adjustment: ", cH.flat[idx])

        # if bit == '1' and rounded_value % 2 != 0:  # If the bit is '1' but the value is odd
        #     cH.flat[idx] += 1
        # elif bit == '0' and rounded_value % 2 == 0:  # If the bit is '0' but the value is even
        #     cH.flat[idx] += 1

    return cH


def create_coeff_pairs(channel, hash_length, quant_table=JPEG_QUANT_TABLE, seed=42):
    np.random.seed(seed)
    # print("Hash length: ", hash_length)
    total_coeffs = channel.size
    coeff_pairs = []

    required_indices = hash_length * 2
    if required_indices > total_coeffs:
        raise ValueError("Not enough coefficients to create required number of pairs")

    # select unique indices upfront
    selected_indices = np.random.choice(total_coeffs, required_indices, replace=False)

    # Pair up the selected indices (resahpe the flat array into a 2-column array for pairs)
    coeff_pairs = selected_indices.reshape(-1, 2)

    return coeff_pairs.tolist()  # convert back to list of tuples


# Function that embeds using Coefficient Pair Differentials
def embed_with_pairs(cH, binary_hash, coeff_pairs, seed=42):
    # np.random.seed(seed)

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


def extract_with_pairs(cH, coeff_pairs, ecc_symbols=20, seed=42):
    np.random.seed(seed)

    binary_hash = ""
    for idx1, idx2 in coeff_pairs:
        diff = round(cH.flat[idx1] - cH.flat[idx2], 2)
        print("Diff: ", diff)
        if diff > 11.0:
            diff = np.floor(diff)
        elif diff < -11.0:
            diff = np.ceil(diff)
        elif -5 < diff < 0:
            diff = np.floor(diff)
        elif 0 < diff < 5:
            diff = np.ceil(diff)

        if 4.5 <= diff <= 11:
            binary_hash += "1"
        elif -11 <= diff <= -4.5:
            binary_hash += "0"

    print("Extracted binary hash: ", binary_hash)
    print("Length of extracted binary hash: ", len(binary_hash))
    # Convert binary to bytes
    hash_bytes = bytes(
        [int(binary_hash[i : i + 8], 2) for i in range(0, len(binary_hash), 8)]
    )
    print("Bytes after conversion", hash_bytes)
    print("Length of bytes after conversion: ", len(hash_bytes))

    # Initialize RS codec and attempt to decode
    rs = RSCodec(ecc_symbols)
    try:
        decoded_hash, _, _ = rs.decode(hash_bytes)
        print("Length of decoded_hash: ", len(decoded_hash))
        if len(decoded_hash) == 32:
            return decoded_hash
        else:
            return None
    except ReedSolomonError as e:
        print("Error decoding hash:", e)
        return None


def extract_segment(channel, coeff_pairs, seed=42):
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


def select_random_blocks(num_blocks, num_to_select, seed=42):
    np.random.seed(seed)
    selected_indices = np.random.choice(num_blocks, num_to_select, replace=False)
    return selected_indices


def quant_table_embed(dct_block, bit_hash, quant_table=JPEG_QUANT_TABLE, threshold=5):
    """
    Function that embeds based off of the jpeg quant table. Takes advantage of the way jpeg image are compressed
    and which information is kept or discarded

    Args :
    --------

    Returns :
    --------
    """

    # Flatten the quantization table and sort indices by quantization value (low to high)
    # Meaning the values that will undergo the least compression will be near the beginning of the array
    # This should work regardless of the absolute values of the quant table
    sorted_indices = np.argsort(quant_table.flatten())
    # print("Sorted indices: ", sorted_indices)

    # Embed the bits into the DCT coefficients with lower frequencies
    for i, bit in enumerate(bit_hash):
        idx = sorted_indices[i]
        if bit == "1":
            dct_block.flat[idx] += threshold * quant_table.flat[idx]
        else:
            dct_block.flat[idx] -= threshold * quant_table.flat[idx]

    return dct_block


def embed_data_in_dct(dct_block, data, quant_table=JPEG_QUANT_TABLE, base_threshold=5):
    sorted_indices = np.argsort(quant_table.flatten())

    for i, bit in enumerate(data):
        idx = sorted_indices[i]
        # Adjust embedding strength based on the quantization coefficient
        threshold = (
            base_threshold * quant_table.flat[idx] / 16
        )  # Example adjustment formula

        if bit == "1":
            dct_block.flat[idx] += (
                threshold * quant_table.flat[idx]
            )  # Set to a high value for '1'
        else:
            dct_block.flat[idx] -= (
                threshold * quant_table.flat[idx]
            )  # Set to a low value for '0'
    return dct_block


def embed_across_blocks(
    y_channel, bit_hash, quant_table=JPEG_QUANT_TABLE, block_size=8
):
    height, width = y_channel.shape
    num_blocks = (height // block_size) * (width // block_size)
    bits_per_block = len(quant_table.flatten())  # Assuming one bit per DCT coefficient
    total_bits = len(bit_hash)
    if total_bits > bits_per_block * num_blocks:
        raise ValueError("Not enough blocks to embed the entire hash")

    for i in range(0, total_bits, bits_per_block):
        row = (i // bits_per_block) // (width // block_size) * block_size
        col = (i // bits_per_block) % (width // block_size) * block_size
        block_data = bit_hash[i : i + bits_per_block]
        block = (
            y_channel[row : row + block_size, col : col + block_size].astype(float)
            - 128
        )
        dct_block = cv2.dct(block)
        dct_block = embed_data_in_dct(dct_block, block_data, quant_table)
        block = cv2.idct(dct_block) + 128
        y_channel[row : row + block_size, col : col + block_size] = np.clip(
            block, 0, 255
        )

    return y_channel


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
