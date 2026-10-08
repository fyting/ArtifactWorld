from PIL import Image

def load_video(video_file):
    from decord import VideoReader

    vr = VideoReader(video_file)
    fps = float(vr.get_avg_fps())
    total_frames = len(vr)
    if total_frames < 1:
        raise RuntimeError(f"Video has no frames: {video_file}")
    if fps <= 0:
        raise RuntimeError(f"Video fps is invalid ({fps}): {video_file}")

    # 10 fps, same index formula as the original loader.
    target_fps = 10
    num_frames_to_extract = int(total_frames * target_fps / fps)
    if num_frames_to_extract < 1:
        frame_indices = [0]
    else:
        frame_indices = []
        for i in range(num_frames_to_extract):
            index = int(i * fps / target_fps)
            if index >= total_frames:
                index = total_frames - 1
            frame_indices.append(index)

    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frames[i]) for i in range(len(frame_indices))]