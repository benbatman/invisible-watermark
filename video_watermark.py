import cv2
from helpers import (
    generate_rc_encoded_hash_from_image,
    select_random_blocks,
    binarize_and_encode_json,
)
from video_utils import read_video_frames, embed_motion
from argparse import ArgumentParser


# Will initially test on MP4 and AVI first
def main(video_path, manifest=None, key_pattern=[2, 0, 3, 1], ecc_symbols=20):
    frames = read_video_frames(video_path)
    print("Length of frames: ", len(frames))
    # print(frames[0])
    # print(frames[0].shape)
    ycrcb_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb_frame[:, :, 0]
    if manifest:
        print("Generating hash from manifest")
        bit_hash, byte_hash, bit_hash_length = binarize_and_encode_json(manifest)
    else:
        encoded_hash, bit_hash, hash_length = generate_rc_encoded_hash_from_image(
            frames[0]
        )

    NUM_BLOCKS = (y_channel.shape[0] // 8) * (y_channel.shape[1] // 8)
    print("Number of blocks: ", NUM_BLOCKS)
    NUM_SELECTED_BLOCKS = int(len(bit_hash) / 4)
    print("Number of selected blocks: ", NUM_SELECTED_BLOCKS)
    selected_blocks = select_random_blocks(NUM_BLOCKS, NUM_SELECTED_BLOCKS)
    # print("Selected blocks: ", selected_blocks)
    segment_length = min(int((len(bit_hash) / 4)) // len(selected_blocks), 8)
    print("Segment length: ", segment_length)

    embedded_frames = embed_motion(
        frames, bit_hash, key_pattern, selected_blocks, segment_length
    )
    print("Length of embedded frames: ", len(embedded_frames))
    height, width = embedded_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    print("Codec: ", fourcc)
    fps = 25

    # Write frames to new video
    out = cv2.VideoWriter("videos/output_3.mp4", fourcc, fps, (width, height))
    if not out.isOpened():
        print("Failed to open video writer.")
        return
    for frame in embedded_frames:
        out.write(frame)

    out.release()


if __name__ == "__main__":
    main("videos/3997798-uhd_2160_4096_25fps.mp4", manifest="test_digi_inno.json")
