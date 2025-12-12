import requests
import time
from volcengine import visual
from volcengine.visual.VisualService import VisualService

import  oss2

import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio.transforms as T
import torch
from config import Config

def generate_vocal(audio_path):
    
    model = get_model(name='htdemucs')
    waveform, sr = torchaudio.load(audio_path)

    # 检查是否是立体声，Demucs 通常使用立体声
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)  # 模拟双声道（复制通道）

    sources = apply_model(model, waveform[None])[0]  # [source, channel, time]
    source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']

    vocals_index = source_names.index('vocals')
    vocals = sources[vocals_index]  # shape: [2, time]

    audio_path_without_extension = audio_path.split(".")[0]
    save_path = f"{audio_path_without_extension}_vocals.wav"
    torchaudio.save(save_path, vocals, sr)
    print(f"Vocals saved to {save_path}")
    
    return save_path

def download_video(url, output_path):

    try:
        response = requests.get(url, stream=True)

        # 确保请求成功
        if response.status_code == 200:
            # 打开输出文件并写入内容
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Video downloaded successfully and saved to {output_path}")
        else:
            print(f"Failed to download video. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading video: {e}")
def upload_to_aliyun(image_path,config):
    """上传到阿里云OSS"""
    # 配置信息
    access_key_id = config.ALIYUN_ID
    access_key_secret = config.ALIYUN_SECRET
    endpoint = 'oss-cn-beijing.aliyuncs.com'
    bucket_name= config.ALIYUN_OSS_BUCKET_NAME
    
    # 创建Bucket对象
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    
    # 上传文件
    file_name = image_path.split('/')[-1]
    object_name = f'{file_name}'  # 存储路径
    with open(image_path, 'rb') as file:
        data = file.read()
    bucket.put_object(object_name, data)
    print(f"File has been uploaded to aliyun OSS：{bucket_name}/{object_name}")
    file_url = bucket.sign_url('GET', object_name, 3600)
    # 返回公网URL
    return file_url

def generate_lip_video_from_jimeng(image_path, audio_path, output_video_path, prompt, config):
    visual_service = VisualService()
    vocal_audio_path = generate_vocal(audio_path)
    # call below method if you don't set ak and sk in $HOME/.volc/config
    visual_service.set_ak('AKLTMmZhZDc2YTk0ZTg2NGM1MWFiZjNmN2Y2ZDNkNWIyZTY')
    visual_service.set_sk('WmpjNU1UazROakZpTTJaaU5ETTJNR0ZtWmpBell6UXhNRGs0WldFMk0yVQ==')
    image_url = upload_to_aliyun(image_path,config)
    audio_url = upload_to_aliyun(vocal_audio_path,config)
    form = {
    "audio_url": f"{audio_url}",
    "image_url": f"{image_url}",
    "req_key": "jimeng_realman_avatar_picture_omni_v15",
    "prompt": prompt
    }
    resp = visual_service.cv_submit_task(form)
    task_id = resp["data"]["task_id"]
    form = {
        "req_key": "jimeng_realman_avatar_picture_omni_v15",
        "task_id": f"{task_id}"
    }
    # 计算用时
    current_time = time.time()
    while True:
        resp = visual_service.cv_get_result(form)
        print(resp)
        status = resp["data"]["status"]
        if status == "done":
            video_url = resp["data"]["video_url"]
            download_video(video_url, output_video_path)
            break
        time.sleep(10)
    now_time = time.time()
    print(f"用时：{now_time - current_time}秒")


def gen_lip_sync_video_jimeng(name: str, config: type = Config):
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
        generate_lip_video_from_jimeng(item["image"], item["audio"], save_path, item["prompt"])
    
        trim_video_to_audio_length(video_path=save_path, audio_path=item["audio"])

if __name__ == '__main__':
    image_path = "/map-vepfs/nicolaus625/m2v/tangxiaoxuan/work/music_pipeline/6/picture1/23/1.jpg"  # Local path to your image
    audio_path = "/map-vepfs/nicolaus625/m2v/tangxiaoxuan/work/music_pipeline/6/piece/23.wav"  # Local path to your audio file
    output_video_path = "23_final.mp4"  # Path where you want the video saved
    generate_lip_video_from_jimeng(image_path, audio_path, output_video_path)
    
    