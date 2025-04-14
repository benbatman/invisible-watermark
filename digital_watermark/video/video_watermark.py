import multiprocessing
import logging
from typing import List

import cv2
import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from tqdm import tqdm

from digital_watermark.video.video_utils import read_video_frames, is_low_motion
from digital_watermark.shared_utils import (
    select_random_blocks,
    create_coeff_pairs,
    embed_with_pairs,
    binarize_and_encode_watermark,
    extract_segment,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VideoWatermark:
    def __init__(
        self,
        watermark: str,
        output_fps: int = 25,
        key_pattern=[2, 0, 3, 1],
        ecc_symbols=20,
    ):
        """
        Args:
            watermark (str): The watermark to embed in the video.
            output_fps (int): The frames per second of the output video.
            key_pattern (list): The pattern used for embedding the watermark.
            ecc_symbols (int): The number of error correction symbols.
        """
        self.watermark = watermark
        self.output_fps = output_fps
        self.key_pattern = key_pattern
        self.ecc_symbols = ecc_symbols

    def embed_watermark(self, video_path: str, output_path: str):
        frames = read_video_frames(video_path)
        ycrcb_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_frame[:, :, 0]
        bit_hash, byte_hash, bit_hash_length = binarize_and_encode_watermark(
            self.watermark, self.ecc_symbols
        )
        NUM_BLOCKS = (y_channel.shape[0] // 8) * (y_channel.shape[1] // 8)
        print("Number of blocks: ", NUM_BLOCKS)
        NUM_SELECTED_BLOCKS = int(len(bit_hash) / 4)
        print("Number of selected blocks: ", NUM_SELECTED_BLOCKS)
        selected_blocks = select_random_blocks(NUM_BLOCKS, NUM_SELECTED_BLOCKS)
        # print("Selected blocks: ", selected_blocks)
        segment_length = min(int((len(bit_hash) / 4)) // len(selected_blocks), 8)
        print("Segment length: ", segment_length)

        # Embed the watermark in the video frames
        embedded_frames = self._embed_motion(
            frames, bit_hash, selected_blocks, segment_length
        )

        height, width = embedded_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # TODO: Add check for different codecs
        self.save_video(output_path, fourcc, width, height, embedded_frames)
        logger.info(f"Watermark embedded in video: {output_path}")
        logger.info(f"Watermark: {self.watermark}")
        logger.info(f"Verifying watermark can be successfully extracted")
        decoded_hash = self.extract_watermark(output_path)
        # Turn decoded_hash into string
        decoded_hash = "".join([format(byte, "08b") for byte in decoded_hash])
        logger.info(f"Decoded hash: {decoded_hash}")
        logger.info(f"Length of decoded hash: {len(decoded_hash)}")
        logger.info(f"Length of original hash: {len(bit_hash)}")
        logger.info(f"Watermark: {self.watermark}")
        if decoded_hash == bit_hash:
            logger.info("Watermark successfully extracted!")

    def _embed_motion(
        self, frames, bit_hash: str, selected_blocks, segment_length
    ) -> List[cv2.typing.MatLike]:

        embedded_frames = [frames[0]]

        # Starts embedding at frame 2 since we calculate the motion vector
        # based on the previous frame and the current frame
        for i in tqdm(range(1, len(frames))):
            prev_frame = frames[i - 1]
            frame = frames[i]
            bit_hash_index = self.key_pattern[i % len(self.key_pattern)]
            # Hash part is a subset of the full bit_hash with length of len(bit_hash) / 4
            hash_part = bit_hash[bit_hash_index :: len(self.key_pattern)]
            embedded_frame = self._embed_with_motion_pairs(
                prev_frame, frame, hash_part, selected_blocks, segment_length
            )
            embedded_frames.append(embedded_frame)

        return embedded_frames

    def _embed_with_motion_pairs(
        self, prev_frame, frame, bit_hash, selected_blocks, segment_length
    ) -> cv2.typing.MatLike:

        low_motion_mask = is_low_motion(prev_frame, frame, 10)
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_frame[:, :, 0]

        h, w = y_channel.shape
        hash_index = 0
        for block_index in selected_blocks:
            if hash_index >= len(bit_hash):
                break

            # Get block's starting row and column based on its index
            row = (block_index // (w // 8)) * 8
            col = (block_index % (w // 8)) * 8

            # Check for low motion frame
            if low_motion_mask[row : row + 8, col : col + 8].mean() > 0.5:
                # Adjust block position based on motion
                # dx, dy = np.mean(flow[row:row+8, col:col+8], axis=(0,1)) # Mean flow vector for the block
                # new_row, new_col = int(row + dy), int(col + dx)

                # # Ensure new block position is within image bounds
                # new_row = min(max(new_row, 0), h-8)
                # new_col = min(max(new_col, 0), w-8)

                current_segment = bit_hash[hash_index : hash_index + segment_length]
                # print("Length of current segment: ", len(current_segment))
                block = (
                    y_channel[row : row + 8, col : col + 8].astype(float) - 128
                )  # Zero shift

                dct_block = cv2.dct(block)
                # Create coefficient pairs from the low frequency parts of the image
                coeff_pairs = create_coeff_pairs(dct_block, len(current_segment))
                dct_block = embed_with_pairs(dct_block, current_segment, coeff_pairs)
                # IDCT
                block = cv2.idct(dct_block) + 128  # Zero shift
                y_channel[row : row + 8, col : col + 8] = np.clip(block, 0, 255).astype(
                    np.uint8
                )
                hash_index += segment_length

            # Need to maintain a hash length of 104 that is extracted
            else:
                current_segment = bit_hash[hash_index : hash_index + segment_length]
                # print("Length of current segment: ", len(current_segment))
                block = (
                    y_channel[row : row + 8, col : col + 8].astype(float) - 128
                )  # Zero shift

                dct_block = cv2.dct(block)
                # Create coefficient pairs from the low frequency parts of the image
                coeff_pairs = create_coeff_pairs(dct_block, len(current_segment))
                dct_block = embed_with_pairs(dct_block, current_segment, coeff_pairs)
                # IDCT
                block = cv2.idct(dct_block) + 128  # Zero shift
                y_channel[row : row + 8, col : col + 8] = np.clip(block, 0, 255).astype(
                    np.uint8
                )
                hash_index += segment_length

        # Update the Y channel in the YCrCb image
        ycrcb_frame[:, :, 0] = y_channel
        # Convert the frame back to BGR color space
        embedded_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)

        return embedded_frame

    def _extract_motion(
        self,
        watermarked_frames,
        key_pattern,
        selected_blocks,
        segment_length,
        bit_hash_length,
    ):

        extracted_hash = {k: [] for k in key_pattern}
        reconstructed_hash = [None] * bit_hash_length
        final_hashes = []

        # Start extraction at frame 2, must be same as embedding
        for i in tqdm(range(1, len(watermarked_frames))):
            # Get the position within the key pattern for this frame (same pattern as embedding)
            pattern_index = i % len(key_pattern)
            # Get the segment index this frame corresponds to
            segment_index = key_pattern[pattern_index]
            prev_frame = watermarked_frames[i - 1]
            frame = watermarked_frames[i]
            bit_hash_index = key_pattern[i % len(key_pattern)]
            # Each time this runs, it will return bits of length 104 since the original hash was split evenly across 4 frames
            extracted_frame_bits = self._extract_with_motion_pairs(
                prev_frame, frame, selected_blocks, segment_length, bit_hash_length / 4
            )
            extracted_hash[bit_hash_index].extend(extracted_frame_bits)

            # Reconstruct the hash
            for j, bit in enumerate(extracted_frame_bits):
                hash_position = (j * len(key_pattern)) + segment_index
                if hash_position < bit_hash_length:
                    reconstructed_hash[hash_position] = bit

            # Every four consecutive frames, join individual hashes into the original
            if i % 4 == 0 or i == len(watermarked_frames) - 1:
                assert (
                    len(reconstructed_hash) == bit_hash_length
                ), "Length of reconstructed hash does not match expected hash length of 416"
                # In case the hash wasn't fully constructed, pad with '0's
                reconstructed_hash = [
                    bit if bit is not None else "0" for bit in reconstructed_hash
                ]
                complete_hash = "".join(reconstructed_hash)
                final_hashes.append(complete_hash)
                reconstructed_hash = [None] * bit_hash_length

        return final_hashes

    def _extract_with_motion_pairs(
        self, prev_frame, frame, selected_blocks, segment_length, bit_hash_length
    ):
        low_motion_mask = is_low_motion(prev_frame, frame, 10)
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_frame[:, :, 0]

        extracted_bits = []
        h, w = y_channel.shape
        hash_index = 0

        for block_index in selected_blocks:
            if hash_index >= bit_hash_length:
                break

            # Calculate block's original position
            row = (block_index // (w // 8)) * 8
            col = (block_index % (w // 8)) * 8
            if low_motion_mask[row : row + 8, col : col + 8].mean() > 0.5:
                block = (
                    y_channel[row : row + 8, col : col + 8].astype(float) - 128
                )  # Zero shift
                dct_block = cv2.dct(block)
                coeff_pairs = create_coeff_pairs(dct_block, segment_length)
                bits = extract_segment(dct_block, coeff_pairs)
                extracted_bits.extend(bits)
                hash_index += segment_length

            else:
                block = (
                    y_channel[row : row + 8, col : col + 8].astype(float) - 128
                )  # Zero shift
                dct_block = cv2.dct(block)
                coeff_pairs = create_coeff_pairs(dct_block, segment_length)
                bits = extract_segment(dct_block, coeff_pairs)
                extracted_bits.extend(bits)
                hash_index += segment_length

        return extracted_bits

    def extract_watermark(self, video_path: str):
        watermarked_frames = read_video_frames(video_path)
        ycrcb_frame = cv2.cvtColor(watermarked_frames[0], cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_frame[:, :, 0]

        if self.watermark:
            bit_str, _, bit_hash_length = binarize_and_encode_watermark(
                self.watermark, self.ecc_symbols
            )

        NUM_BLOCKS = (y_channel.shape[0] // 8) * (y_channel.shape[1] // 8)
        print("Number of blocks: ", NUM_BLOCKS)
        NUM_SELECTED_BLOCKS = int(bit_hash_length / 4)
        print("Number of selected blocks: ", NUM_SELECTED_BLOCKS)
        selected_blocks = select_random_blocks(NUM_BLOCKS, NUM_SELECTED_BLOCKS)
        # print("Selected blocks: ", selected_blocks)
        segment_length = min(int(bit_hash_length / 4) // len(selected_blocks), 8)
        print("Segment length: ", segment_length)

        rs = RSCodec(self.ecc_symbols)

        extracted_hashes = self._extract_motion(
            watermarked_frames,
            self.key_pattern,
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

            try:
                decoded_hash, decoded_hash_and_ec, _ = rs.decode(hash_bytes)
                print("Length of decoded_hash: ", len(decoded_hash))

                if len(decoded_hash_and_ec) == bit_hash_length / 8:
                    print("Huzzah, hash successfully decoded!")
                    print("Decoded hash: ", decoded_hash)
                    return decoded_hash

            except Exception as e:
                logger.error(f"Error decoding hash with RSCodec: {e}")

        logger.error("Failed to decode the watermark hash.")

    def save_video(self, output_path: str, fourcc, width, height, frames):
        out = cv2.VideoWriter(output_path, fourcc, self.output_fps, (width, height))
        if not out.isOpened():
            logger.error("Failed to open video writer.")
            return
        for frame in frames:
            out.write(frame)

        out.release()
