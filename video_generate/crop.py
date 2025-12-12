# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import logging
import os
import os.path as osp
import shutil
import subprocess

import imageio
import torch
import torchvision
import numpy as np

__all__ = ['save_video', 'save_image', 'str2bool']


import cv2
import os
import tempfile
import shutil
import json

def crop_video(input_file, output_file, sec):
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 判断输入和输出是否为同一路径
    same_path = os.path.abspath(input_file) == os.path.abspath(output_file)

    # 如果是同一路径，则创建临时输出文件
    if same_path:
        # 创建临时文件
        temp_fd, temp_output_file = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)  # 关闭文件描述符
    else:
        temp_output_file = output_file

    try:
        # 设置输入视频的文件名和裁剪时间段
        filename = input_file
        # 打开视频文件
        cap = cv2.VideoCapture(filename)

        # 获取视频的帧率、总帧数和时长
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 将时间戳转换为秒数
        start_sec = 0
        end_sec = sec

        # 计算裁剪时间段的起始帧和结束帧
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        # 设置裁剪后输出视频的文件名和编码器
        output_filename = temp_output_file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 设置输出视频的帧率和分辨率
        out_fps = fps
        out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建输出视频的对象
        out = cv2.VideoWriter(output_filename, fourcc, out_fps, (out_width, out_height))

        # 跳转到裁剪时间段的起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 读取并写入裁剪时间段内的每一帧
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break

        # 释放对象并关闭窗口
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 如果是同一路径，将临时文件移动到目标位置
        if same_path:
            shutil.move(temp_output_file, output_file)

    except Exception as e:
        # 如果发生异常，清理临时文件
        if same_path and os.path.exists(temp_output_file):
            os.remove(temp_output_file)
        raise e

def crop_lip_video(song_name):
    video_folder_path = f"/map-vepfs/nicolaus625/m2v/tangxiaoxuan/work/{song_name}/output1/lip_sync"
    video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]
    json_path = f"/map-vepfs/nicolaus625/m2v/tangxiaoxuan/work/{song_name}/camera1"
    for video_file in video_files:
        # print(video_file)
        file_name = video_file.split('.')[0]
        json_file = os.path.join(json_path, f'{file_name}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
            duration = data[0]['shot_duration']
            crop_video(os.path.join(video_folder_path, video_file), os.path.join(video_folder_path, video_file), duration)


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    # set logging
    logging.basicConfig(level=logging.INFO)

    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # overwrite
            '-i',
            video_path,
            '-i',
            audio_path,
            '-c:v',
            'copy',  # copy video stream
            '-c:a',
            'aac',  # use AAC audio encoder
            '-b:a',
            '192k',  # set audio bitrate (optional)
            '-map',
            '0:v:0',  # select the first video stream
            '-map',
            '1:a:0',  # select the first audio stream
            '-shortest',  # choose the shortest duration
            temp_output
        ]

        # execute the command
        logging.info("Start merging video and audio...")
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        logging.info(f"Merge completed, saved to {video_path}")

    except Exception as e:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        logging.error(f"merge_video_audio failed with error: {e}")


def trim_video_to_audio_length(video_path: str, audio_path: str, fps: float = 24.0):
    import librosa
    import cv2
    
    # logging setup
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_duration = librosa.get_duration(path=audio_path)
    logging.info(f"Audio duration: {audio_duration:.3f}s")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = fps  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps

    logging.info(f"Video duration: {video_duration:.3f}s, FPS: {video_fps}")

    target_frame_count = int(np.floor(audio_duration * video_fps))
    logging.info(f"Target frame count: {target_frame_count} / {total_frames}")

    base, ext = os.path.splitext(video_path)
    trimmed_path = f"{base}_trimmed{ext}"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(trimmed_path, fourcc, video_fps, (width, height))

    frame_idx = 0
    while frame_idx < target_frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    logging.info(f"Trimmed video saved: {trimmed_path}")

    final_path = f"{base}_final{ext}"
    cmd = [
        'ffmpeg', '-y',
        '-i', trimmed_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        final_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    os.remove(trimmed_path)
    logging.info(f"Final merged video: {final_path}")

    return final_path


def save_video(tensor,
               save_file=None,
               fps=30,
               suffix='.mp4',
               nrow=8,
               normalize=True,
               value_range=(-1, 1)):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    try:
        # preprocess
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack([
            torchvision.utils.make_grid(
                u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor.unbind(2)
        ],
                             dim=1).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        # write video
        writer = imageio.get_writer(
            cache_file, fps=fps, codec='libx264', quality=8)
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        logging.info(f'save_video failed, error: {e}')


def save_image(tensor, save_file, nrow=8, normalize=True, value_range=(-1, 1)):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    try:
        tensor = tensor.clamp(min(value_range), max(value_range))
        torchvision.utils.save_image(
            tensor,
            save_file,
            nrow=nrow,
            normalize=normalize,
            value_range=value_range)
        return save_file
    except Exception as e:
        logging.info(f'save_image failed, error: {e}')


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')


def masks_like(tensor, zero=False, generator=None, p=0.2):
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[:, 0] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[:, 0]).exp()
                    v[:, 0] = torch.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]
        else:
            for u, v in zip(out1, out2):
                u[:, 0] = torch.zeros_like(u[:, 0])
                v[:, 0] = torch.zeros_like(v[:, 0])

    return out1, out2


def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio)**0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2,
                                                 ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2


def download_cosyvoice_repo(repo_path):
    try:
        import git
    except ImportError:
        raise ImportError('failed to import git, please run pip install GitPython')
    repo = git.Repo.clone_from('https://github.com/FunAudioLLM/CosyVoice.git', repo_path, multi_options=['--recursive'], branch='main')


def download_cosyvoice_model(model_name, model_path):
    from modelscope import snapshot_download
    snapshot_download('iic/{}'.format(model_name), local_dir=model_path)
