from helpers import (
    generate_rc_encoded_hash_from_image,
    select_random_blocks,
    create_coeff_pairs_jpg,
    embed_with_pairs,
    extract_segment,
)
import numpy as np
import cv2
from reedsolo import RSCodec, ReedSolomonError
from argparse import ArgumentParser


def main(image_upload_path, image_download_path, ecc_symbols=20):
    image = cv2.imread(image_upload_path)
    print("Image shape: ", image.shape)
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    encoded_hash, bit_hash, hash_length = generate_rc_encoded_hash_from_image(image)

    y_channel = ycrcb_image[:, :, 0]

    NUM_BLOCKS = (y_channel.shape[0] // 8) * (y_channel.shape[1] // 8)
    print("Number of blocks: ", NUM_BLOCKS)
    NUM_SELECTED_BLOCKS = len(bit_hash)
    print("Number of selected blocks: ", NUM_SELECTED_BLOCKS)
    selected_blocks = select_random_blocks(NUM_BLOCKS, NUM_SELECTED_BLOCKS)
    # print("Selected blocks: ", selected_blocks)

    segment_length = min(len(bit_hash) // len(selected_blocks), 8)
    print("Segment length: ", segment_length)

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
            coeff_pairs = create_coeff_pairs_jpg(dct_block, len(current_segment))
            dct_block = embed_with_pairs(dct_block, current_segment, coeff_pairs)

            # IDCT
            block = cv2.idct(dct_block) + 128  # Zero shift
            y_channel[row : row + 8, col : col + 8] = np.clip(block, 0, 255).astype(
                np.uint8
            )

            hash_index += segment_length

    # Update the Y channel in the YCrCb image
    ycrcb_image[:, :, 0] = y_channel
    # Convert the YCrCb image back to RGB
    watermarked_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)
    # Save the image
    cv2.imwrite(image_download_path, watermarked_image)

    # Load the watermarked image and try to extract the hash
    watermarked_image = cv2.imread(image_download_path)
    ycrcb_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb_image[:, :, 0]
    print("Y channel shape: ", y_channel.shape)

    NUM_BLOCKS = (y_channel.shape[0] // 8) * (y_channel.shape[1] // 8)
    print("Number of blocks: ", NUM_BLOCKS)
    NUM_SELECTED_BLOCKS = len(bit_hash)
    print("Number of selected blocks: ", NUM_SELECTED_BLOCKS)
    selected_blocks = select_random_blocks(NUM_BLOCKS, NUM_SELECTED_BLOCKS)
    # print("Selected blocks: ", selected_blocks)

    segment_length = min(len(bit_hash) // len(selected_blocks), 8)
    print("Segment length: ", segment_length)

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
            coeff_pairs = create_coeff_pairs_jpg(dct_block, len(current_segment))
            extracted_segment = extract_segment(dct_block, coeff_pairs)
            extracted_hash += extracted_segment
            hash_index += segment_length

    print("Extracted hash: ", extracted_hash)
    print("Length of extracted hash: ", len(extracted_hash))
    hash_bytes = bytes(
        [int(extracted_hash[i : i + 8], 2) for i in range(0, len(extracted_hash), 8)]
    )
    print("Length of hash bytes: ", len(hash_bytes))
    print("Hash bytes: ", hash_bytes)

    rs = RSCodec(ecc_symbols)
    try:
        decoded_hash, _, _ = rs.decode(hash_bytes)
        print("Length of decoded_hash: ", len(decoded_hash))
        if len(decoded_hash) == 32:
            print("Huzzah, hash successfully decoded!")
    except ReedSolomonError as e:
        print("Error decoding hash:", e)


if __name__ == "__main__":
    parser = ArgumentParser(description="Process an image for watermarking")
    parser.add_argument(
        "--upload_path", type=str, required=True, help="Path to upload image to process"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="watermarked_image.jpg",
        help="Path to output image",
    )
    args = parser.parse_args()

    main(args.upload_path, args.output_path)
