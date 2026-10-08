#!/usr/bin/env python3
"""Label video artifacts with the vendored Q-Align checkpoint.

Examples
--------
Single video:
    python tools/infer_q_align.py --video /path/to/clip.mp4

A list of videos, one path per line:
    python tools/infer_q_align.py --video_list examples/q_align_video_list.txt --output labels.json

Multi-GPU data parallel (one full model replica per GPU):
    python tools/infer_q_align.py --video_list examples/q_align_video_list.txt --output labels.json --num_gpus 8
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QALIGN_WEIGHTS_ROOT = Path(
    os.environ.get("QALIGN_WEIGHTS_ROOT", REPO_ROOT / "q_align_weights")
)

from q_align.artifacts import ARTIFACTS
from q_align.predictor import ArtifactPredictor


def default_model_path():
    return str(QALIGN_WEIGHTS_ROOT / "q-align-artifacts")


def default_model_base():
    return str(QALIGN_WEIGHTS_ROOT / "one-align")


def read_video_list(path):
    videos = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                videos.append(line)
    return videos


def entry_from_result(video_path, result):
    return {
        "video": video_path,
        "artifacts": {
            name: {
                "prediction": item["prediction"],
                "prob_yes": item["prob_yes"],
                "prob_no": item["prob_no"],
            }
            for name, item in result.items()
        },
    }


def shard_path(output, gpu_id):
    if output.endswith(".json"):
        return output[:-5] + f"_gpu{gpu_id}.json"
    return output + f"_gpu{gpu_id}.json"


def worker_inference(gpu_id, video_paths, args, return_dict):
    local_output = shard_path(args.output, gpu_id)
    results = []
    processed = set()
    if os.path.exists(local_output):
        try:
            with open(local_output, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            if isinstance(saved, list):
                results = saved
            elif isinstance(saved, dict) and "results" in saved:
                results = saved["results"]
            processed = {item["video"] for item in results}
            print(f"[GPU {gpu_id}] resume {len(results)} videos from {local_output}")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[GPU {gpu_id}] ignore unreadable resume file {local_output}: {exc}")
            results = []
            processed = set()

    predictor = ArtifactPredictor(
        args.model_path,
        args.model_base,
        device=f"cuda:{gpu_id}",
        use_8bit=args.use_8bit,
        max_frames=args.max_frames,
    )

    pending = [path for path in video_paths if path not in processed]
    for index, video_path in enumerate(tqdm(pending, desc=f"GPU {gpu_id}", position=gpu_id), start=1):
        if not os.path.exists(video_path):
            tqdm.write(f"[GPU {gpu_id}] missing file: {video_path}")
            continue
        try:
            result = predictor.predict(video_path, artifacts=args.artifacts)
        except Exception as exc:
            tqdm.write(f"[GPU {gpu_id}] failed {video_path}: {exc}")
            continue
        results.append(entry_from_result(video_path, result))
        if len(results) % args.save_every == 0:
            with open(local_output, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2, ensure_ascii=False)

    with open(local_output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    return_dict[gpu_id] = len(results)
    print(f"[GPU {gpu_id}] wrote {len(results)} videos to {local_output}")


def merge_shards(output, num_gpus, total_videos):
    merged = []
    for gpu_id in range(num_gpus):
        local_output = shard_path(output, gpu_id)
        if not os.path.exists(local_output):
            continue
        with open(local_output, "r", encoding="utf-8") as handle:
            merged.extend(json.load(handle))
    payload = {
        "total_videos": total_videos,
        "successful": len(merged),
        "results": merged,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(merged)} results to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Video artifact labeling")
    parser.add_argument("--video", type=str, default=None, help="Single video path")
    parser.add_argument("--video_list", type=str, default=None, help="Text file of video paths")
    parser.add_argument("--output", type=str, default="labels.json", help="Output JSON")
    parser.add_argument("--model_path", type=str, default=default_model_path())
    parser.add_argument("--model_base", type=str, default=default_model_base())
    parser.add_argument(
        "--artifacts",
        nargs="+",
        default=None,
        choices=ARTIFACTS,
        help="Subset of artifacts. Default: all nine.",
    )
    parser.add_argument("--max_frames", type=int, default=60)
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.video) == bool(args.video_list):
        raise SystemExit("Pass exactly one of --video or --video_list")
    mp.set_start_method("spawn", force=True)

    if args.video:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required.")
        predictor = ArtifactPredictor(
            args.model_path,
            args.model_base,
            device="cuda:0",
            use_8bit=args.use_8bit,
            max_frames=args.max_frames,
        )
        result = predictor.predict(args.video, artifacts=args.artifacts)
        payload = entry_from_result(args.video, result)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"Saved {args.output}")
        return

    videos = read_video_list(args.video_list)
    if not videos:
        raise SystemExit(f"No videos in {args.video_list}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    available = torch.cuda.device_count()
    num_gpus = available if args.num_gpus is None else min(args.num_gpus, available)
    if num_gpus < 1:
        raise SystemExit("No GPU detected.")
    print(f"{len(videos)} videos, using {num_gpus}/{available} GPUs")

    if num_gpus == 1:
        args.num_gpus = 1
        manager = mp.Manager()
        return_dict = manager.dict()
        worker_inference(0, videos, args, return_dict)
        merge_shards(args.output, 1, len(videos))
        return

    chunk_size = math.ceil(len(videos) / num_gpus)
    chunks = [videos[i : i + chunk_size] for i in range(0, len(videos), chunk_size)]
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    started = time.time()
    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        process = mp.Process(target=worker_inference, args=(gpu_id, chunk, args, return_dict))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise SystemExit(f"Worker exited with code {process.exitcode}")
    print(f"Workers finished in {time.time() - started:.1f}s")
    merge_shards(args.output, num_gpus, len(videos))


if __name__ == "__main__":
    main()
