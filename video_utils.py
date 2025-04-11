import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers import (
    perform_dwt,
    perform_idwt,
    create_coeff_pairs_jpg,
    embed_with_pairs,
    select_random_blocks,
    extract_segment,
)
import multiprocessing
from tqdm import tqdm


# Function to read video frames
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    return frames


def get_codec(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()
    return codec


def generate_pn_sequence(length, seed=42):
    np.random.seed(seed)
    return np.random.choice([-1, 1], size=length)


def is_high_motion(prev_frame, current_frame, threshold=50000):
    """
    Determine if there is high motion between two consecutive frames
    The threshold may need adjustment based on video resolution and desired sensitivity
    Frames must be grayscale.
    """

    # Comput the absolute difference between the two frames
    frame_diff = cv2.absdiff(current_frame, prev_frame)

    # Threshold the difference to get a binary image
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Sum up the values of all pixels to represent the amount of motion
    motion_level = np.count_nonzero(thresh)

    # If the sum is above the threshold, there is high motion
    return motion_level > threshold


def is_complex_frame(frame, edges_threshold=150000):
    """
    Determine if a frame is complex based on number of edges
    Adjust threshold based on video resolution and desired sensitivity
    """

    # Use Canny edge detection to detect edges
    edges = cv2.Canny(frame, 100, 200)

    # Count the number of edges
    num_edges = np.count_nonzero(edges)

    return num_edges > edges_threshold


def analyze_frame(frame, frame_idx, prev_gray):
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if is_high_motion(prev_gray, current_gray) or is_complex_frame(current_gray):
        return frame_idx, True
    else:
        return frame_idx, False


def find_frames(video_path, skip_frames=5, batch_size=10):
    """
    Find frames with high motion or complex content. This method is used for two reasons:

    1. In video compression, frames
    """

    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        print("Failed to read video")
        cap.release()
        return

    frame_idx = 0
    embed_frame_indices = []  # Hold all frame indices
    high_motion_complex_frames = []

    while True:
        batch_frames = []
        batch_indices = []

        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to speed up processing
            frame_idx += skip_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            batch_frames.append(frame)
            batch_indices.append(frame_idx)

            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        with ThreadPoolExecutor(
            max_workers=(multiprocessing.cpu_count() / 2)
        ) as executor:
            results = executor.map(
                lambda p: analyze_frame(*p, prev_gray), zip(batch_frames, batch_indices)
            )

        for result, frame in zip(results, batch_frames):
            frame_idx, is_high_motion_or_complex = result
            if is_high_motion_or_complex:
                embed_frame_indices.append(frame_idx)
                high_motion_complex_frames.append(frame)

        if not ret:
            break

    cap.release()
    return high_motion_complex_frames, embed_frame_indices


def embed_frame(frame, watermark, strength=0.01):
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Split the channels
    y = ycrcb_frame[:, :, 0]
    # Apply DWT to the Y channel
    cA, cH, cV, cD = perform_dwt(y.astype(np.float64))
    watermark_resized = np.resize(watermark, cA.shape)
    cA_watermarked = cA + strength * watermark_resized
    reconstructed_frame_y = perform_idwt(cA_watermarked, cH, cV, cD)
    # print("Max and min values of frame: ", max(reconstructed_frame_y.flatten()), min(reconstructed_frame_y.flatten()))
    # Ensure values are within pixel value range
    reconstructed_frame_y = np.clip(reconstructed_frame_y, 0, 255).astype(np.uint8)
    # Update the Y channel in the YCrCb frame
    ycrcb_frame[:, :, 0] = reconstructed_frame_y
    # Convert the frame back to BGR color space
    frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)

    return frame


def embed_parallel(frames, watermark, strength=0.01):
    embedded_frames = []
    with ThreadPoolExecutor() as executor:
        # Future object for each frame
        futures = [
            executor.submit(embed_frame, frame, watermark, strength) for frame in frames
        ]

        for future in as_completed(futures):
            embedded_frames.append(future.result())

    return embedded_frames


def embed_frames(frames, bit_hash, key_pattern, strength=0.01):
    embedded_frames = []
    # Convert frame to YCrCb color space
    for i, frame in enumerate(frames):
        bit_hash_index = key_pattern[i % len(key_pattern)]
        watermark = bit_hash[bit_hash_index :: len(key_pattern)]
        ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Split the channels
        y_channel = ycrcb_frame[:, :, 0]

        # # Apply DWT to the Y channel
        # cA, cH, cV, cD = perform_dwt(y.astype(np.float64))
        # watermark_resized = np.resize(watermark, cA.shape)
        # cA_watermarked = cA + strength * watermark_resized
        # reconstructed_frame_y = perform_idwt(cA_watermarked, cH, cV, cD)
        # print("Max and min values of frame: ", max(reconstructed_frame_y.flatten()), min(reconstructed_frame_y.flatten()))

        NUM_BLOCKS = (y_channel.shape[0] // 8) * (y_channel.shape[1] // 8)
        # print("Number of blocks: ", NUM_BLOCKS)
        NUM_SELECTED_BLOCKS = len(watermark)
        # print("Number of selected blocks: ", NUM_SELECTED_BLOCKS)
        selected_blocks = select_random_blocks(NUM_BLOCKS, NUM_SELECTED_BLOCKS)
        # print("Selected blocks: ", selected_blocks)

        segment_length = min(len(watermark) // len(selected_blocks), 8)
        # print("Segment length: ", segment_length)

        # Process the Y channel in 8x8 blocks
        h, w = y_channel.shape
        hash_index = 0
        for row in range(0, h, 8):
            for col in range(0, w, 8):
                if hash_index >= len(watermark):
                    break

                current_segment = watermark[hash_index : hash_index + segment_length]
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
        ycrcb_frame[:, :, 0] = y_channel

        # Convert the frame back to BGR color space
        frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)

        embedded_frames.append(frame)

    return embedded_frames


def reconstruct_video(
    video_path, high_motion_complex_frames, embed_frame_indices, output_path
):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    original_idx = 0
    embed_idx = 0
    last_embedded_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if original_idx in embed_frame_indices:
            # Replace the current frame with the corresponding embedded frame
            frame = high_motion_complex_frames[embed_idx]
            embed_idx += 1
            last_embedded_frame = frame
        elif last_embedded_frame is not None:
            frame = last_embedded_frame

        out.write(frame)
        original_idx += 1

    cap.release()
    out.release()


####### Motion Embedding ########


def is_low_motion(prev_frame, frame, threshold):

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)

    # Threshold the difference to get a binary image, 1 represents low motion, 0 represents motion
    low_motion_mask = np.where(diff < threshold, 1, 0).astype(np.uint8)

    return low_motion_mask


def estimate_motion(prev_frame, frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # flow variable is a 2D array of optical flow vectors (u, v) which are horizontal and vertical displacement
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    return flow


def embed_with_motion_compensation(prev_frame, frame, flow, hash_part):
    u, v = flow.shape[:2]
    flow_map = np.column_stack((flow[..., 1].flatten(), flow[..., 0].flatten()))
    positions = []

    for i in range(len(hash_part)):
        positions.append((i % u, i // v))

    new_positions = []
    for x, y in positions:
        dx, dy = flow[y, x]
        new_x, new_y = int(x + dx), int(y + dy)
        new_positions.append((new_x % u, new_y % v))

    return frame


def embed_with_motion_pairs(
    prev_frame, frame, bit_hash, selected_blocks, segment_length
):
    # flow = estimate_motion(prev_frame, frame)
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
            coeff_pairs = create_coeff_pairs_jpg(dct_block, len(current_segment))
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
            coeff_pairs = create_coeff_pairs_jpg(dct_block, len(current_segment))
            dct_block = embed_with_pairs(dct_block, current_segment, coeff_pairs)
            # IDCT
            block = cv2.idct(dct_block) + 128  # Zero shift
            y_channel[row : row + 8, col : col + 8] = np.clip(block, 0, 255).astype(
                np.uint8
            )
            hash_index += segment_length

    # print("Length of hash embedded: ", hash_index)
    # Update the Y channel in the YCrCb image
    ycrcb_frame[:, :, 0] = y_channel
    # Convert the frame back to BGR color space
    embedded_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR)

    return embedded_frame


def embed_motion(frames, bit_hash, key_pattern, selected_blocks, segment_length):
    embedded_frames = [frames[0]]

    # Starts embedding at frame 2
    for i in tqdm(range(1, len(frames))):
        prev_frame = frames[i - 1]
        frame = frames[i]
        bit_hash_index = key_pattern[i % len(key_pattern)]
        # Hash part is a subset of the full bit_hash with length of len(bit_hash) / 4
        hash_part = bit_hash[bit_hash_index :: len(key_pattern)]
        # print("Hash part to embed: ", hash_part)
        # print("Length of hash part: ", len(hash_part))
        embedded_frame = embed_with_motion_pairs(
            prev_frame, frame, hash_part, selected_blocks, segment_length
        )
        embedded_frames.append(embedded_frame)

    return embedded_frames


def extract_with_motion_pairs(
    prev_frame, frame, selected_blocks, segment_length, bit_hash_length
):
    # flow = estimate_motion(prev_frame, frame)
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

            # Adjust block position based on reverse action
            # dx, dy = np.mean(flow[row:row+8, col:col+8], axis=(0, 1))
            # org_row, org_col = int(row - dy), int(col - dx)

            # # Ensure original block position is within image bounds
            # org_row = min(max(org_row, 0), h-8)
            # org_col = min(max(org_col, 0), w-8)

            block = (
                y_channel[row : row + 8, col : col + 8].astype(float) - 128
            )  # Zero shift
            dct_block = cv2.dct(block)
            coeff_pairs = create_coeff_pairs_jpg(dct_block, segment_length)
            bits = extract_segment(dct_block, coeff_pairs)
            extracted_bits.extend(bits)
            hash_index += segment_length

        else:
            block = (
                y_channel[row : row + 8, col : col + 8].astype(float) - 128
            )  # Zero shift
            dct_block = cv2.dct(block)
            coeff_pairs = create_coeff_pairs_jpg(dct_block, segment_length)
            bits = extract_segment(dct_block, coeff_pairs)
            extracted_bits.extend(bits)
            hash_index += segment_length

    return extracted_bits


def extract_motion(frames, key_pattern, selected_blocks, segment_length, hash_length):
    extracted_hash = {k: [] for k in key_pattern}
    reconstructed_hash = [None] * hash_length
    final_hashes = []

    # Start extraction at frame 2, must be same as embedding
    for i in tqdm(range(1, len(frames))):
        # Get the position within the key pattern for this frame (same pattern as embedding)
        pattern_index = i % len(key_pattern)
        # Get the segment index this frame corresponds to
        segment_index = key_pattern[pattern_index]
        prev_frame = frames[i - 1]
        frame = frames[i]
        bit_hash_index = key_pattern[i % len(key_pattern)]
        # Each time this runs, it will return bits of length 104 since the original hash was split evenly across 4 frames
        extracted_frame_bits = extract_with_motion_pairs(
            prev_frame, frame, selected_blocks, segment_length, hash_length / 4
        )
        # print("length of extracted frames bits: ", len(extracted_frame_bits))
        extracted_hash[bit_hash_index].extend(extracted_frame_bits)

        # Reconstruct the hash
        for j, bit in enumerate(extracted_frame_bits):
            hash_position = (j * len(key_pattern)) + segment_index
            if hash_position < hash_length:
                reconstructed_hash[hash_position] = bit

        # Every four consecutive frames, join individual hashes into the original
        if i % 4 == 0 or i == len(frames) - 1:
            assert (
                len(reconstructed_hash) == hash_length
            ), "Length of reconstructed hash does not match expected hash length of 416"
            # In case the hash wasn't fully constructed, pad with '0's
            reconstructed_hash = [
                bit if bit is not None else "0" for bit in reconstructed_hash
            ]
            complete_hash = "".join(reconstructed_hash)
            final_hashes.append(complete_hash)
            reconstructed_hash = [None] * hash_length

    # assert len(reconstructed_hash) == hash_length, "Length of reconstructed hash does not match expected hash length of 416"
    # reconstructed_hash = [bit if bit is not None else '0' for bit in reconstructed_hash]
    # final_hashes.append(reconstructed_hash)
    # for i in range(hash_length): # Total number of bits to reconstruct (104 * 4 = 416)
    #     key_index = i % len(key_pattern)
    #     segment_index = key_pattern[key_index]
    #     frame_index = (i // len(key_pattern)) * len(key_pattern) + segment_index  # Calculate the frame index this bit would have been embedded in
    #     bit_index_within_segment = i // len(key_pattern)

    #     # Check if the frame and segment have enough bits extracted
    #     if frame_index < len(frames) and bit_index_within_segment < len(extracted_hash[segment_index]):
    #         reconstructed_hash.append(extracted_hash[segment_index][bit_index_within_segment])

    # part_index = key_pattern[i % len(key_pattern)]
    # bit_index = i // len(key_pattern)
    # print("Part index: ", part_index)
    # print("Bit index: ", bit_index)
    # if bit_index < len(extracted_hash[part_index]):
    #     reconstructed_hash.append(extracted_hash[part_index][bit_index])
    #     # print("Reconstructed hash: ", reconstructed_hash)

    # print("Length of reconstructed hash: ", len(reconstructed_hash))

    return final_hashes


def reconstruct_hash(frames, key_pattern, segment_length=104):
    hash_segments = {k: [] for k in range(len(key_pattern))}

    # Iterate through frames, extracting hash segments according to the key pattern
    for i, frame in enumerate(frames):
        if i % len(key_pattern) == 0 and i > 0:
            # Every four frames, attempt to reconstruct the hash if enough data has been collected
            reconstructed_hash = []
            for segment_index in sorted(hash_segments.keys()):
                reconstructed_hash.extend(hash_segments[segment_index])

            # Enusre the reconstructed hash is the expected length before returning it
            if len(reconstructed_hash) == segment_length * len(key_pattern):
                return reconstructed_hash
