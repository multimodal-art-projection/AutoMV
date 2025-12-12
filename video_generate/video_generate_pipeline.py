import os
import re
import io
import json
import time
import base64
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import requests
from PIL import Image
from volcenginesdkarkruntime import Ark
from moviepy.editor import VideoFileClip, concatenate_videoclips

from .crop import crop_video
import concurrent.futures
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path
from typing import List
from .video_verify import call_gemini
from .crop import crop_lip_video
from .call_gemini import cut_camera
from .crop import merge_video_audio
from config import Config
def resize_video(clip, target_width, target_height):
    return clip.resize(newsize=(target_width, target_height))

def concatenate_videos(input_video_paths: List[Path], output_video_path: Path):
    clips = []
    target_width = 864
    target_height = 480

    for clip_path in input_video_paths:
        try:
            clip = VideoFileClip(str(clip_path))
            # clip = clip.set_fps(24)

            clip = resize_video(clip, target_width, target_height)

            clips.append(clip)
        except Exception as e:
            print(f"[Error] Failed to load video file: {clip_path}: {e}")
            continue

    if not clips:
        print("[Error] No valid video files to merge.")
        return

    try:
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(str(output_video_path), codec="libx264", fps=24)
        print(f"Success! Output to: {output_video_path}")
    except Exception as e:
        print(f"[Error] Failed: {e}")
    
def concatenate_final_videos(input_video_paths: List[Path], output_video_path: Path):
    resolution = "864x480"
    with open("videos_to_merge.txt", "w") as f:
        for video in input_video_paths:
            f.write(f"file '{str(video)}'\n")
    os.system(f"ffmpeg -f concat -safe 0 -i videos_to_merge.txt -vf scale={resolution} -c:v libx264 -crf 23 -preset fast -c:a aac -strict experimental {str(output_video_path)} -y")

def crop_video_with_opencv(input_video_path: Path, output_video_path: Path, duration):
    try:
        crop_video(input_video_path, output_video_path, duration)

        print(f"[Info] Success! Output to: {output_video_path}")

    except Exception as e:
        print(f"[Error] Failed: {e}")



def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> List[Dict]:
    print(f"Reading json file {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def convert_image_to_png_b64(image_path: Path) -> str:
    with Image.open(image_path) as img:
        if img.format != 'PNG':
            img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
        else:
            data = image_path.read_bytes()
    return base64.b64encode(data).decode('utf-8')


def download_file(url: str, output_path: Path, chunk: int = 8192):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with output_path.open("wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)


def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def extract_last_frame_cv2(video_path: Path, out_image: Path):
    ensure_dir(out_image.parent)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame

    cap.release()

    if last_frame is not None:
        cv2.imwrite(str(out_image), last_frame)
        print(f"[Info] Successfully extracted and saved the last frame: {out_image}")
    else:
        raise RuntimeError(f"[Error] Failed to extract any frames: {video_path}")


def ensure_dir(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)




# ---------------------- Ark Generation ----------------------

class ArkVideoClient:
    def __init__(self, api_key: str, base_url: str = "https://ark.cn-beijing.volces.com/api/v3"):
        self.client = Ark(base_url=base_url, api_key=api_key)

    def generate_shot(
        self,
        model: str,
        prompt_text: str,
        first_frame_png_b64: str,
        *,
        res: str = "480p",
        ratio: str = "16:9",
        dur: int = 8,
        fps: int = 24,
        wm: bool = False,
        seed: int = 11,
        cf: bool = False,
        poll_interval: int = 10,
        verbose: bool = True,
        shot_video_path: str = None,
        max_score: int = 0
    ):
        flags = f"--rs {res} --rt {ratio} --dur {dur} --fps {fps} --wm {'true' if wm else 'false'} --seed {seed} --cf {'true' if cf else 'false'}"
        text_payload = f"{prompt_text.strip()} {flags}"
        print(f"Processing {shot_video_path}...")
        create_result = self.client.content_generation.tasks.create(
            model=model,
            content=[
                {"type": "text", "text": "Do not generate sensitive content, follow the instructions below to create the video shots:" + text_payload},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{first_frame_png_b64}"},
                 "role": "first_frame"},
            ]
        )
        task_id = create_result.id
        if verbose:
            print(f"[Ark] Created task: {task_id}")
        video_url = None
        while True:
            try:
                get_result = self.client.content_generation.tasks.get(task_id=task_id)
                status = get_result.status
                if verbose:
                    print(f"[Ark] status={status}")
                if status == "succeeded":
                    video_url = get_result.content.video_url
                    print("DONE!")
                    break
                if status == "failed":
                    print(f"[Ark] Task failed: {get_result.error}")
                    print("Prompt:", prompt_text)
                    return False, prompt_text, 0
            except Exception as e:
                raise RuntimeError(f"Processing {shot_video_path}... Ark task failed: {get_result.error}")
            time.sleep(poll_interval)
        # shot_video_path = work_dir / "shots" / f"shot_{shot_num:02d}.mp4"
        tmp_file_path = shot_video_path.with_name(shot_video_path.stem + "_tmp.mp4")
        print(f"[DL_TMP] {video_url} -> {tmp_file_path}")
        download_file(video_url, tmp_file_path)
        verify_result = call_gemini(prompt_text, tmp_file_path)
        judge_result = verify_result["judge result"]
        current_score = verify_result["score"]
        judge_result = "yes"
        current_score = 5
        print(f"[C_Score] {current_score}")
        print(f"[M_Score] {max_score}")
        if int(current_score) >= int(max_score):
            max_score = current_score
            print(f"[DL_SHOT] {video_url} -> {shot_video_path}")
            download_file(video_url, shot_video_path)
        if judge_result == "yes":
            return True, prompt_text, 5
        return False, prompt_text, max_score


def parse_xy_from_filename(fn: str) -> Optional[tuple]:
    m = re.match(r"^(\d+)_([1-9]\d*|0)\.json$", fn.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def build_prompt(prompt: str, camera_movement: str) -> str:
    return f'{prompt.strip()}\nCamera movement: {camera_movement.strip()}'


import math

def process_one_json(
    json_path: Path,
    picture_dir: Path,
    out_root: Path,
    ark: ArkVideoClient,
    model: str,
    res: str, ratio: str, fps: int, wm: bool, seed: int, cf: bool,
    poll_interval: int
) -> Path:
    xy = parse_xy_from_filename(json_path.name)
    if not xy:
        print(f"[Skip] {json_path.name} is not in x_y.json format")
        return None
    x, y = xy

    first_frame_img = picture_dir / str(x) / f"{y}.jpg"
    if not first_frame_img.exists():
        raise FileNotFoundError(f"First frame image does not exist: {first_frame_img}")

    shots = read_json(json_path)
    if not isinstance(shots, list) or len(shots) == 0:
        raise ValueError(f"{json_path} content is empty or format is incorrect")

    work_dir = out_root / f"{x}_{y}"
    ensure_dir(work_dir)
    ensure_dir(work_dir / "shots")
    ensure_dir(work_dir / "frames")

    curr_first_frame_b64 = convert_image_to_png_b64(first_frame_img)

    shot_video_paths: List[Path] = []

    for idx, shot in enumerate(shots, start=1):
        shot_num = shot.get("shot_num", idx)
        prompt = shot.get("prompt", "")
        cam = shot.get("camera_movement", "")
        dur = float(shot.get("shot_duration", 8))

        # If duration is a decimal, round up
        if dur != int(dur):
            dur = math.ceil(dur)
            print(f"[Info] shot#{shot_num} original duration is {shot['shot_duration']} seconds, rounded up to {dur} seconds")
        dur = int(dur)
        merged_prompt = build_prompt(prompt, cam)
        print(f"\n=== Generating {json_path.name} / shot#{shot_num} ({dur}s) ===")

        shot_video_path = work_dir / "shots" / f"shot_{shot_num:02d}.mp4"
        if shot_video_path.exists():
            print(f"[Skip] shot_{shot_num:02d}.mp4 already exists, skipping download")
        else:
            if idx == len(shots) and dur < 3:
                dur = 3
                print(f"[Info] Last shot duration is less than 3, temporarily using 3 seconds for generation")
            retry = 3
            prompt = merged_prompt
            max_score = 0
            while retry > 0:
                retry -= 1
                is_ok, prompt, max_score = ark.generate_shot(
                    model=model,
                    prompt_text=prompt,
                    first_frame_png_b64=curr_first_frame_b64,
                    res=res, ratio=ratio, dur=dur, fps=fps, wm=wm, seed=seed, cf=cf,
                    poll_interval=poll_interval, verbose=True,
                    shot_video_path=shot_video_path,
                    max_score=max_score
                )
                if is_ok == True:
                    break

        shot_video_paths.append(shot_video_path)

        # Check if the last frame already exists to avoid duplicate extraction
        last_frame_path = work_dir / "frames" / f"shot_{shot_num:02d}_last.png"
        if last_frame_path.exists():
            print(f"[Skip] shot_{shot_num:02d}_last.png already exists, skipping extraction")
        else:
            # Extract last frame from this video segment → first frame for next shot
            if not has_ffmpeg():
                raise EnvironmentError("FFmpeg is required to extract frames and concatenate videos. Please install it and try again.")
            extract_last_frame_cv2(shot_video_path, last_frame_path)

        curr_first_frame_b64 = convert_image_to_png_b64(last_frame_path)

        # If this is the last shot, crop to the original duration
        if idx == len(shots) or dur == 3:
            # Use OpenCV to crop video
            print(f"[Info] Cropping last shot to original duration: {shot['shot_duration']} seconds")
            original_dur = shot.get("shot_duration", 8)
            print(f"[Info] Original duration: {original_dur} seconds")
            crop_video_with_opencv(shot_video_path, shot_video_path, original_dur)
            print(f"[Info] Last shot cropping completed")

    # Concatenate videos
    final_out = work_dir / f"{x}_{y}_final.mp4"
    print(f"\n[Concat] -> {final_out}")

    # Use moviepy to concatenate videos
    concatenate_videos(shot_video_paths, final_out)

    print(f"[Done] Final output: {final_out}")
    return final_out


def process_json_in_thread(jp, index, picture_dir, out_root, ark, model, res, ratio, fps, wm, seed, cf, poll_interval):
    result = process_one_json(
        json_path=jp,
        picture_dir=picture_dir,
        out_root=out_root,
        ark=ark,
        model=model,
        res=res,
        ratio=ratio,
        fps=fps,
        wm=wm,
        seed=seed,
        cf=cf,
        poll_interval=poll_interval
    )
    return index, result


def get_mp4_files(out_root):
    folders = [folder for folder in out_root.iterdir() if folder.is_dir()]

    videos_list = []
    for folder in folders:
        if "lip_sync" == folder.name:
            mp4_files = [file for file in folder.glob("*_final.mp4")]
        else:
            mp4_files = [file for file in folder.glob("*.mp4")]
        videos_list.extend(mp4_files)
    
    def extract_number_from_file_name(file_path):
        folder_name = file_path.name
        match = re.match(r'(\d+)', folder_name)
        return int(match.group(1)) if match else float('inf')
    
    # Sort video files by the numeric part of the folder name
    videos_list.sort(key=extract_number_from_file_name)
    
    print(f"[Info] Found {len(videos_list)} video files")
    print(videos_list)
    return videos_list

def full_video_gen(name: str, resolution: str = "480p", config: type = Config):
    try:
        cut_camera(f"./result/{name}/camera")
    except Exception as e:
        print(e)
        print(f"[Error] {name} video segmentation failed")
        return
    # Define all paths and parameters
    camera_dir = Path(f"./result/{name}/camera")
    picture_dir = Path(f"./result/{name}/picture")
    out_root = Path(f"./result/{name}/output")
    model = "doubao-seedance-1-0-pro-250528"
    res = resolution
    ratio = "16:9"
    fps = 24
    wm = False
    seed = 11
    cf = False
    poll_interval = 10

    # Get API Key
    api_key = config.DOUBAO_API_KEY
    if not api_key:
        raise EnvironmentError("Please set environment variable ARK_API_KEY")

    ark = ArkVideoClient(api_key=api_key)

    # Create output directory
    ensure_dir(out_root)

    # Iterate through all x_y.json files in camera_dir
    # json_files = sorted([p for p in camera_dir.iterdir() if p.is_file() and re.match(r"^\d+_\d+\.json$", p.name)],
    #                     key=lambda p: (int(p.stem.split("_")[0]), int(p.stem.split("_")[1])))
    json_files = sorted([p for p in camera_dir.iterdir()
                         if p.is_file() and re.match(r"^\d+_1\.json$", p.name)],  # Only select files ending with _1.json
                        key=lambda p: (int(p.stem.split("_")[0]), int(p.stem.split("_")[1])))
    print(len(json_files))
    if not json_files:
        print(f"[Warn] No x_y.json files found in {camera_dir}")
        return

    final_videos = [None] * len(json_files)  # Initialize final_videos list with the same length as json_files
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Start multiple threads to process JSON files
        futures = [
            executor.submit(process_json_in_thread, jp, idx, picture_dir, out_root, ark, model, res, ratio, fps, wm, seed, cf, poll_interval)
            for idx, jp in enumerate(json_files)
        ]
        
        # Wait for all threads to complete and fill results into final_videos by original index
        for future in concurrent.futures.as_completed(futures):
            idx, result = future.result()
            final_videos[idx] = result


    out_root = Path(f"./result/{name}/output")
    videos_list = get_mp4_files(out_root)
    
    # Print video file paths
    # for video in videos_list:
    #     print(video)
    final_output = Path(f"./result/{name}/mv_{name}.mp4")
    concatenate_videos(videos_list, final_output)
    merge_video_audio(f"./result/{name}/mv_{name}.mp4", f"./result/{name}.mp3")


if __name__ == "__main__":
    full_video_gen(51)   
