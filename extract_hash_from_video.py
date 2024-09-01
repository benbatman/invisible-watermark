import numpy as np
import cv2
from helpers import (
    generate_rc_encoded_hash_from_image,
    select_random_blocks,
    binarize_and_encode_json,
)
from video_utils import read_video_frames, extract_motion
from argparse import ArgumentParser
from reedsolo import RSCodec, ReedSolomonError

# Extraction requires knowledge of the key pattern and the starting parameters of the RSCodec
# Without key pattern, there is no way to accurately reconstruct the binary hash


def main(video_path, manifest=None, key_pattern=[2, 0, 3, 1], ecc_symbols=20):
    watermarked_frames = read_video_frames(video_path)
    print("Length of frames: ", len(watermarked_frames))
    # Get constants
    ycrcb_frame = cv2.cvtColor(watermarked_frames[0], cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb_frame[:, :, 0]

    if manifest:
        _, _, bit_hash_length = binarize_and_encode_json(manifest)

    NUM_BLOCKS = (y_channel.shape[0] // 8) * (y_channel.shape[1] // 8)
    print("Number of blocks: ", NUM_BLOCKS)
    NUM_SELECTED_BLOCKS = int(bit_hash_length / 4)
    print("Number of selected blocks: ", NUM_SELECTED_BLOCKS)
    selected_blocks = select_random_blocks(NUM_BLOCKS, NUM_SELECTED_BLOCKS)
    # print("Selected blocks: ", selected_blocks)
    segment_length = min(int(bit_hash_length / 4) // len(selected_blocks), 8)
    print("Segment length: ", segment_length)

    # extracted_hash = extract_motion(watermarked_frames, key_pattern, selected_blocks, segment_length)
    # extracted_hash = ''.join(extracted_hash)

    # print("Extracted hash: ", extracted_hash)
    # print("Length of extracted hash: ", len(extracted_hash))
    # hash_bytes = bytes([int(extracted_hash[i:i+8], 2) for i in range(0, len(extracted_hash), 8)])
    # print("Length of hash bytes: ", len(hash_bytes))
    # print("Hash bytes: ", hash_bytes)

    rs = RSCodec(ecc_symbols)

    extracted_hashes = extract_motion(
        watermarked_frames,
        key_pattern,
        selected_blocks,
        segment_length,
        bit_hash_length,
    )
    for extracted_hash in extracted_hashes:
        # print("Extracted hash: ", extracted_hash)
        # print("Length of extracted hash: ", len(extracted_hash))
        hash_bytes = bytes(
            [
                int(extracted_hash[i : i + 8], 2)
                for i in range(0, len(extracted_hash), 8)
            ]
        )
        # print("Length of hash bytes: ", len(hash_bytes))
        # print("Hash bytes: ", hash_bytes)

        try:
            decoded_hash, decoded_hash_and_ec, _ = rs.decode(hash_bytes)
            print("Length of decoded_hash: ", len(decoded_hash))
            if len(decoded_hash_and_ec) == bit_hash_length / 8:
                print("Huzzah, hash successfully decoded!")
                print("Decoded hash: ", decoded_hash)
                break
        except:
            continue

        finally:
            print("Could not find c2pa manifest in media")


if __name__ == "__main__":
    main("videos/3997798-uhd_2160_4096_25fps.mp4", manifest="test_digi_inno.json")
