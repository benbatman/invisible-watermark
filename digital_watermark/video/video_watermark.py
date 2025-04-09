import multiprocessing

import cv2
import numpy as np
from tqdm import tqdm

from video_utils import read_video_frames, is_low_motion, binarize_and_encode_watermark
from digital_watermark.shared_utils import (
    select_random_blocks,
    create_coeff_pairs,
    embed_with_pairs,
)


class VideoWatermark:
    def __init__(self, watermark: str, key_pattern=[2, 0, 3, 1]):
        self.watermark = watermark
        self.key_pattern = key_pattern

    def embed_watermark(self, video_path: str):
        frames = read_video_frames(video_path)
        ycrcb_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb_frame[:, :, 0]
        bit_hash, byte_hash, bit_hash_length = binarize_and_encode_watermark(
            self.watermark
        )
        NUM_BLOCKS = (y_channel.shape[0] // 8) * (y_channel.shape[1] // 8)
        print("Number of blocks: ", NUM_BLOCKS)
        NUM_SELECTED_BLOCKS = int(len(bit_hash) / 4)
        print("Number of selected blocks: ", NUM_SELECTED_BLOCKS)
        selected_blocks = select_random_blocks(NUM_BLOCKS, NUM_SELECTED_BLOCKS)
        # print("Selected blocks: ", selected_blocks)
        segment_length = min(int((len(bit_hash) / 4)) // len(selected_blocks), 8)
        print("Segment length: ", segment_length)

    def _embed_motion(self, frames, bit_hash, selected_blocks, segment_length):
        embedded_frames = [frames[0]]

        # Starts embedding at frame 2
        for i in tqdm(range(1, len(frames))):
            prev_frame = frames[i - 1]
            frame = frames[i]
            bit_hash_index = self.key_pattern[i % len(self.key_pattern)]
            # Hash part is a subset of the full bit_hash with length of len(bit_hash) / 4
            hash_part = bit_hash[bit_hash_index :: len(self.key_pattern)]
            # print("Hash part to embed: ", hash_part)
            # print("Length of hash part: ", len(hash_part))
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
