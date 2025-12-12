import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio.transforms as T
import torch
import os
def generate_vocal(audio_path):
    audio_path_without_extension = audio_path.split(".")[0]
    save_path = f"{audio_path_without_extension}_vocals.wav"
    save_path = f"{audio_path_without_extension}_vocals.wav"
    if os.path.exists(save_path):
        print(f"{save_path} 已存在，跳过")
        return save_path
    model = get_model(name='htdemucs')
    waveform, sr = torchaudio.load(audio_path)
    
    # 检查是否是立体声，Demucs 通常使用立体声
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)  # 模拟双声道（复制通道）

    sources = apply_model(model, waveform[None])[0]  # [source, channel, time]
    source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']

    vocals_index = source_names.index('vocals')
    vocals = sources[vocals_index]  # shape: [2, time]
    torchaudio.save(save_path, vocals, sr)
    print(f"Vocals saved to {save_path}")
    
    return save_path

from pathlib import Path

def process_mp3_files_modern(base_dir, process_func):
    """
    使用pathlib的现代写法
    """
    base_path = Path(base_dir)
    
    # 查找所有数字文件夹下的数字.mp3文件
    mp3_files = []
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            mp3_file = folder / f"{folder.name}.mp3"
            if mp3_file.exists():
                mp3_files.append(mp3_file)
    
    # 按数字排序
    mp3_files.sort(key=lambda x: int(x.parent.name))
    
    print(f"找到 {len(mp3_files)} 个MP3文件")
    
    # 逐个处理
    for mp3_path in mp3_files:
        print(f"处理: {mp3_path}")
        try:
            result = process_func(str(mp3_path))
            print(f"✓ 完成: {mp3_path.name}")
        except Exception as e:
            print(f"✗ 失败: {mp3_path.name} - {e}")

# 使用示例
#process_mp3_files_modern("/map-vepfs/nicolaus625/m2v/tangxiaoxuan/work/music_pipeline", generate_vocal)
