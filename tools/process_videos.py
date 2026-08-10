#!/usr/bin/env python3
"""Merge Hugging Face benchmark videos (gt + artifact) into reference clips for LTX validation."""

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from tqdm import tqdm


def resolve_ffmpeg():
    for key in ("FFMPEG", "FFMPEG_BINARY"):
        path = os.environ.get(key)
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise FileNotFoundError(
            "ffmpeg not found on PATH; install ffmpeg or imageio-ffmpeg, "
            "or set FFMPEG to an absolute binary path"
        ) from e


def write_video_ffmpeg(output_path, frames, fps, width, height):
    cmd = [
        resolve_ffmpeg(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-vcodec",
        "h264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.read().decode(errors="replace"))


def read_frame_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def process_video(gt_dir, artifact_dir, output_dir, filename):
    gt_path = os.path.join(gt_dir, filename)
    artifact_path = os.path.join(artifact_dir, filename)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        return filename, "skipped"

    cap_gt = cv2.VideoCapture(gt_path)
    if not cap_gt.isOpened():
        return filename, "error: cannot open gt"

    total_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))
    gt_first = read_frame_at(cap_gt, 0)
    gt_last = read_frame_at(cap_gt, total_gt - 1)
    cap_gt.release()

    if gt_first is None or gt_last is None:
        return filename, "error: cannot read gt frames"

    cap_art = cv2.VideoCapture(artifact_path)
    if not cap_art.isOpened():
        return filename, "error: cannot open artifact"

    fps = cap_art.get(cv2.CAP_PROP_FPS)
    width = int(cap_art.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_art.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_art = int(cap_art.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_art < 1:
        cap_art.release()
        return filename, "error: artifact has no frames"

    middle_frames = []
    for i in range(total_art):
        ret, frame = cap_art.read()
        if not ret:
            break
        if i == 0 or i == total_art - 1:
            continue
        middle_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap_art.release()

    all_frames = [gt_first] + middle_frames + [gt_last] + [gt_last] * 8

    try:
        write_video_ffmpeg(output_path, all_frames, fps, width, height)
    except Exception as e:
        return filename, f"error: ffmpeg failed: {e}"

    return filename, "ok"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-dir", required=True, help="Directory of ground-truth mp4 (from gt.tar)")
    p.add_argument("--artifact-dir", required=True, help="Directory of artifact mp4 (from artifact.tar)")
    p.add_argument("--output-dir", required=True, help="Directory to write merged reference mp4")
    p.add_argument("--workers", type=int, default=16, help="Thread pool size")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    filenames = sorted(os.listdir(args.artifact_dir))
    filenames = [f for f in filenames if f.endswith(".mp4")]
    filenames = [f for f in filenames if os.path.exists(os.path.join(args.gt_dir, f))]

    if not filenames:
        print("No matching .mp4 pairs found (artifact + gt).", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(filenames)} videos...")

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_video, args.gt_dir, args.artifact_dir, args.output_dir, f): f
            for f in filenames
        }
        for future in tqdm(as_completed(futures), total=len(filenames)):
            fname, status = future.result()
            if status not in ("ok", "skipped"):
                errors.append((fname, status))

    print(f"\nDone. {len(filenames) - len(errors)} succeeded, {len(errors)} errors.")
    for fname, err in errors:
        print(f"  {fname}: {err}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
