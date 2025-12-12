# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import warnings
from datetime import datetime
import json
import re
warnings.filterwarnings('ignore')
from tqdm import tqdm
import random

import torch
import torch.distributed as dist
from PIL import Image

from . import wan
from .wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from .wan.distributed.util import init_distributed_group
from .wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from .wan.utils.utils import merge_video_audio, save_video, str2bool, trim_video_to_audio_length
from .speech_enhance import generate_vocal
import types
from argparse import Namespace


EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "animate-14B": {
        "prompt": "视频中的人在做动作",
        "video": "",
        "pose": "",
        "mask": "",
    },
    "s2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
        "audio":
            "examples/talk.wav",
        "tts_prompt_audio":
            "examples/zero_shot_prompt.wav",
        "tts_prompt_text":
            "希望你以后能够做的比我还好呦。",
        "tts_text":
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
    if args.audio is None and args.enable_tts is False and "audio" in EXAMPLE_PROMPT[args.task]:
        args.audio = EXAMPLE_PROMPT[args.task]["audio"]
    if (args.tts_prompt_audio is None or args.tts_text is None) and args.enable_tts is True and "audio" in EXAMPLE_PROMPT[args.task]:
        args.tts_prompt_audio = EXAMPLE_PROMPT[args.task]["tts_prompt_audio"]
        args.tts_prompt_text = EXAMPLE_PROMPT[args.task]["tts_prompt_text"]
        args.tts_text = EXAMPLE_PROMPT[args.task]["tts_text"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    if not 's2v' in args.task:
        assert args.size in SUPPORTED_SIZES[
            args.
            task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")

    # animate
    parser.add_argument(
        "--src_root_path",
        type=str,
        default=None,
        help="The file of the process output path. Default None.")
    parser.add_argument(
        "--refert_num",
        type=int,
        default=77,
        help="How many frames used for temporal guidance. Recommended to be 1 or 5."
    )
    parser.add_argument(
        "--replace_flag",
        action="store_true",
        default=False,
        help="Whether to use replace.")
    parser.add_argument(
        "--use_relighting_lora",
        action="store_true",
        default=False,
        help="Whether to use relighting lora.")
    
    # following args only works for s2v
    parser.add_argument(
        "--num_clip",
        type=int,
        default=None,
        help="Number of video clips to generate, the whole video will not exceed the length of audio."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to the audio file, e.g. wav, mp3")
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice to synthesis audio")
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s.")
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Text wish to synthesize")
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Provide Dw-pose sequence to do Pose Driven")
    parser.add_argument(
        "--start_from_ref",
        action="store_true",
        default=False,
        help="whether set the reference image as the starting point for generation"
    )
    parser.add_argument(
        "--infer_frames",
        type=int,
        default=80,
        help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v"
    )
    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    if "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )
        logging.info(f"Generating video ...")
        video = wan_s2v.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=args.num_clip,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.infer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            init_first_frame=args.start_from_ref,
        )
    else: 
        logging.error(f"Only s2v-* tasks supported for this version, got: {args.task}")
        raise ValueError("Only support speech-to-video for lip-sync generation")
   

    return video
    
    
def _process_prompt_input(prompt):
    """
    if prompt is a json file
    """
    if prompt is None:
        print("No prompt given. Use blank prompt input")
        return ""

    if isinstance(prompt, str) and prompt.endswith(".json") and os.path.exists(prompt):
        try:
            with open(prompt, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list) and len(data) > 0:
                entry = data[0]
                text = entry.get("prompt", "")
            elif isinstance(data, dict):
                text = data.get("prompt", "")
            else:
                raise ValueError("Invalid JSON format: must be list or dict.")

            text = text.strip().replace("\n", " ")
            sentences = re.split(r'[.!;]', text)
            first_sentence = sentences[0].strip() if sentences else text

            print(f"[INFO] Extracted first sentence from JSON prompt:\n{first_sentence}")
            return first_sentence

        except Exception as e:
            print(f"[WARN] Failed to parse prompt JSON ({prompt}): {e}")
            return prompt  # 回退为原始字符串

    # 普通字符串 prompt
    return prompt    
    
def generate_video(
    image,
    audio,
    prompt,
    task="s2v-14B",
    size="480*832",
    ckpt_dir="./Wan2.2-S2V-14B/",
    offload_model=True,
    convert_model_dtype=True,
    enable_tts=False,
    save_file=None,
    **kwargs
): 
    """
    Only support single GPU now
    Only need to input four args for basic use:
        size (str)   : video size, choose from 
                 ['720*1280', '1280*720', '480*832', '832*480', '1024*704', '704*1024', '704*1280', '1280*704']
        prompt (str) : text description of the scene, 
                e.g. a stunning young woman with long straight black hair, standing against a soft blue gradient background.
        image (str) : path to input image (.jpg)
        audio (str) : path to input audio (.wav / .mp3)
        
    Return the video tensors
    """
    prompt = _process_prompt_input(prompt)
   
    args = Namespace(
        task=task,
        size=size,
        frame_num=None,
        ckpt_dir=ckpt_dir,
        offload_model=offload_model,
        ulysses_size=1,
        t5_fsdp=False,
        t5_cpu=False,
        dit_fsdp=False,
        save_file=save_file,
        prompt=prompt,
        use_prompt_extend=False,
        prompt_extend_method="local_qwen",
        prompt_extend_model=None,
        prompt_extend_target_lang="zh",
        base_seed=-1,
        image=image,
        sample_solver="unipc",
        sample_steps=None,
        sample_shift=None,
        sample_guide_scale=None,
        convert_model_dtype=convert_model_dtype,
        src_root_path=None,
        refert_num=77,
        replace_flag=False,
        use_relighting_lora=False,
        num_clip=None,
        audio=audio,
        enable_tts=enable_tts,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        pose_video=None,
        start_from_ref=False,
        infer_frames=80,
        **kwargs
    )
    
    # 调用你的验证函数（它会自动补全默认值）
    _validate_args(args)
    # 调用主生成函数
    video = generate(args)
    
    return video


def generate_video_with_clean_vocal(
    image,
    audio,
    prompt,
    task="s2v-14B",
    size="480*832",
    ckpt_dir="./Wan2.2-S2V-14B/",
    offload_model=True,
    convert_model_dtype=True,
    enable_tts=False,
    save_file=None,
    **kwargs
): 
    """
    Only support single GPU now
    Only need to input four args for basic use:
        size (str)   : video size, choose from 
                 ['720*1280', '1280*720', '480*832', '832*480', '1024*704', '704*1024', '704*1280', '1280*704']
        prompt (str) : text description of the scene, 
                e.g. a stunning young woman with long straight black hair, standing against a soft blue gradient background.
        image (str) : path to input image (.jpg)
        audio (str) : path to input audio (.wav / .mp3)
        
    Return the video tensors
    """
    prompt = _process_prompt_input(prompt)
    vocal_path = generate_vocal(audio)
   
    args = Namespace(
        task=task,
        size=size,
        frame_num=None,
        ckpt_dir=ckpt_dir,
        offload_model=offload_model,
        ulysses_size=1,
        t5_fsdp=False,
        t5_cpu=False,
        dit_fsdp=False,
        save_file=save_file,
        prompt=prompt,
        use_prompt_extend=False,
        prompt_extend_method="local_qwen",
        prompt_extend_model=None,
        prompt_extend_target_lang="zh",
        base_seed=-1,
        image=image,
        sample_solver="unipc",
        sample_steps=None,
        sample_shift=None,
        sample_guide_scale=None,
        convert_model_dtype=convert_model_dtype,
        src_root_path=None,
        refert_num=77,
        replace_flag=False,
        use_relighting_lora=False,
        num_clip=None,
        audio=vocal_path,
        enable_tts=enable_tts,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        pose_video=None,
        start_from_ref=False,
        infer_frames=80,
        **kwargs
    )
    
    # 调用你的验证函数（它会自动补全默认值）
    _validate_args(args)
    
    # 调用主生成函数
    video = generate(args)
    
    return video


def gen_lip_sync_video(name: str):
    print(f"[INFO] Processing{name}")
    prompt_folder = f"./result/{name}/camera"
    image_folder = f"./result/{name}/picture"
    audio_folder = f"./result/{name}/piece"
    output_folder = f"./result/{name}/output/lip_sync"
    prompt_list = os.listdir(prompt_folder)
    assets_paths = []
    for prompt in prompt_list:
        with open(os.path.join(prompt_folder, prompt), "r", encoding="utf-8") as f:
            data = json.load(f)[0]
            print(type(data), data)
            if isinstance(data,list): 
                data=data[0]
            label = data.get("label", "")

            if label == "sing":
                filename_without_extension = os.path.splitext(os.path.basename(prompt))[0]
                
                prompt_file_path = os.path.join(prompt_folder, prompt)
                image_file_path = os.path.join(image_folder, filename_without_extension, "1.jpg")
                audio_file_path = os.path.join(audio_folder, filename_without_extension + ".wav")
                
                assets_paths.append({
                    "id": filename_without_extension,
                    "prompt": prompt_file_path,
                    "image": image_file_path,
                    "audio": audio_file_path,
                })
    for item in tqdm(assets_paths,desc="Processing videos"):
        id = item["id"]
        save_path = os.path.join(output_folder, id + ".mp4")
        if os.path.exists(save_path):
            continue
        video = generate_video_with_clean_vocal(
            size = "480*832",
            image = item["image"],
            audio = item["audio"],
            prompt = item["prompt"],
        )
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        save_video(tensor=video[None],save_file=save_path,fps=24,nrow=1,normalize=True,value_range=(-1, 1))
    
        trim_video_to_audio_length(video_path=save_path, audio_path=item["audio"])

if __name__ == "__main__":
    import Config
    gen_lip_sync_video(Config.music_name)