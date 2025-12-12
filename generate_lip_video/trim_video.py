import subprocess
import librosa

video_path = "test16_gen_cam_move5.mp4"
audio_path = "/map-vepfs/nicolaus625/m2v/tangxiaoxuan/work/music/光年之外/16.wav"
output_path = "video_synced.mp4"

# 获取音频精确时长
audio_duration = librosa.get_duration(path=audio_path)

# 使用 ffmpeg 精确裁剪视频时长
cmd = [
    "ffmpeg", "-y",
    "-i", video_path,
    "-t", f"{audio_duration:.6f}",   # 保留到毫秒级
    "-c:v", "copy", "-an",           # 不保留旧音频
    "trimmed_video.mp4"
]
subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 再重新合成
cmd2 = [
    "ffmpeg", "-y",
    "-i", "trimmed_video.mp4",
    "-i", audio_path,
    "-c:v", "copy", "-c:a", "aac",
    "-shortest",
    output_path
]
subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("✅ 已精确裁剪并合成到:", output_path)
