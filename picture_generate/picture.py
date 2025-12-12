import soundfile as sf
import requests
import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from volcenginesdkarkruntime import Ark
from picture_generate.gemini_verify import call_gemini 
from faster_whisper import WhisperModel
from openai import OpenAI
from openai import OpenAIError 
from config import Config

Config.validate()
client_doubao = Ark(
    api_key=Config.DOUBAO_API_KEY
)
client_gemini = OpenAI(
    api_key=Config.GEMINI_API_KEY
)

device = Config.GPU_ID
model_path = Config.qwen_model
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map=device)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

def generate_image_doubao(prompt,save_folder,best_dict):
    imagesResponse = client_doubao.images.generate(
                model="doubao-seedream-4-0-250828",
                prompt=prompt,
                extra_body={"watermark": False,"size": "1920x1080"}
            )
    image_url = imagesResponse.data[0].url
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  

    image_name = "1.jpg" 
    image_path = os.path.join(save_folder, image_name)

    response = requests.get(image_url)
    if response.status_code == 200:
        with open(image_path, 'wb') as file:
            file.write(response.content)
        print(f"图片已成功保存为：{image_path}")
    else:
        print(f"下载图片失败，状态码：{response.status_code}")
    verify_result = call_gemini(prompt,image_path)
    judge_result = verify_result["judge result"]
    score = verify_result["score"]
    if score >= best_dict["score"]:
        best_dict = {
            "score": score,
            "image_url": image_url
        }
    if judge_result == "yes":
      return True, best_dict
    else:
      return False, best_dict
    
def chat_with_gemini(question: str, api_key: str = None, max_retries=5, retry_delay=5) -> str:
    """
    使用 Gemini-2.5 Pro 进行问答
    
    Args:
        question: 用户的问题
        api_key: Google AI API Key，如果不传则从环境变量读取
    
    Returns:
        模型的回答
    """
    retries = 0
    while retries < max_retries:
        try:
            response = client_gemini.chat.completions.create(
                model="gemini-2.5-pro",
                messages=[{"role": "user", "content": question}],
                timeout=1800
            )
            return response.choices[0].message.content

        except OpenAIError as e:
            retries += 1
            print(f"Request failed ({retries}/{max_retries}): {e}")
            time.sleep(retry_delay)
    
    raise RuntimeError(f"Failed to get response after {max_retries} retries")

def qwen_api(audio_files):
    style_before="""
    {
      "Song Description": "The song features a melancholic yet serene melody, evoking traditional Chinese aesthetics. The lyrics weave a narrative of timeless longing, beauty, and destiny, centered around the creation of a blue and white porcelain vase and a deep, unfulfilled love. The song's mood is reflective, romantic, and deeply poetic, with a classical, elegant feel.",
      "Gender": "Male"
    }
    """
    style_prompt = f"""
    Analyze the uploaded song and provide a detailed description that includes its sentiment, themes, genre, mood, and the instruments used. 
    Use both the lyrics and the audio for your analysis, but do not include a transcription of the lyrics in your output.

    Based on this analysis, recommend a single, cohesive visual concept for a music video (MV) production.

    In the "Gender" field, specify the performer’s gender as one of the following: "Male", "Female", "Chorus", or "None".

    Output the results strictly in the following JSON format. 
    "This is an example. Please generate content that is realistic and directly based on the provided audio, avoiding excessive overlap with the example, and focusing on the unique characteristics of the audio itself."
    {style_before}

    Use English for all JSON values, while keeping the keys in English.
    """
    conversations = []
    for audio_path in audio_files:
        conv = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": style_prompt}
                ]
            }
        ]
        conversations.append(conv)

    # ========== 模型输入预处理 ==========
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # ========== 推理 ==========
    text_ids = model.generate(**inputs, use_audio_in_video=False, return_audio=False)
    text_outputs = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # ========== 保存结果到变量 ==========
    results = {}
    for conv, answer in zip(conversations, text_outputs):
        audio_path = [x["audio"] for x in conv[-1]["content"] if x["type"] == "audio"][0]
        filename = os.path.basename(audio_path)
        results[filename] = answer

    indexed_results = {}

    for name, ans in results.items():
        match = re.search(r'\d+', name)
        if "assistant" in ans:
            assistant_reply = ans.split("assistant")[-1].strip()
        if match:
            num = int(match.group())
            indexed_results[num] = assistant_reply
    return indexed_results

def transcribe_audio(audio_path, 
                     model_path=Config.whipser_model,
                     device="cuda", 
                     device_index=0,
                     compute_type="float16",
                     language="zh",
                     beam_size=2):
    """
    使用 faster-whisper 进行音频转录，返回带时间戳的字幕列表。

    参数:
        audio_path (str): 音频文件路径
        model_path (str): 模型路径
        device (str): 使用的设备 ('cuda' 或 'cpu')
        device_index (int): GPU编号
        compute_type (str): 推理精度 ('float16', 'int8', 'float32'等)
        language (str): 语言代码，如 'zh'
        beam_size (int): beam search大小

    返回:
        List[str]: 形如 "[开始时间 - 结束时间] 文本" 的字幕列表
    """
    # 加载模型
    model = WhisperModel(model_path, device=device, device_index=device_index, compute_type=compute_type)

    # 执行转录
    segments, info = model.transcribe(audio_path, beam_size=beam_size)

    # 整理输出结果
    subtitles = [f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}" for segment in segments]

    return subtitles

# 读取JSON文件
def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSON格式错误 - {e}")
        return None
    
def generate_ffmpeg_commands(input_file, story, name):
    commands = []
    for segment in story:
        start_time = segment["start"] 
        duration = segment["end"] - segment["start"]
        
        # 将秒转换为 HH:MM:SS 格式
        hours = int(start_time // 3600)
        minutes = int((start_time % 3600) // 60)
        seconds = start_time % 60
        
        # 格式化时间字符串
        start_str = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
        save_folder = f"./result/{name}/piece"  # 指定文件夹名
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)  # 如果文件夹不存在，则创建它
        output_name = f"{segment['number']}.wav"  # 按序号生成文件名
        output_file = os.path.join(save_folder, output_name)
        # 生成FFmpeg命令
        cmd = f"ffmpeg -y -i {input_file} -ss {start_str} -t {duration:.2f} -f wav {output_file}"
        
        commands.append({
            "number": segment["number"],
            "command": cmd,
            "text": segment["text"]
        })
    
    return commands

def extract_json_from_response(response_json):

    cleaned_text = response_json.strip()

    # Step 2️⃣ 尝试提取 ```json ... ``` 或 ``` ... ``` 代码块
    match = re.search(r'```json\s*(.*?)\s*```', cleaned_text, re.DOTALL)
    if not match:
        match = re.search(r'```(.*?)```', cleaned_text, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
    else:
        raise ValueError("未找到 JSON 代码块，请检查模型输出:\n" + cleaned_text)

    # Step 3️⃣ 尝试解析 JSON
    try:
        json_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"模型输出不是有效的 JSON，解析失败：{e}\n原始文本：\n{json_str}")

    return json_data
  
def generate(para,label,qwen_result,name,try_number=1):
    # picture
    if para["label"] == "sing":
        Gender = "Primary directive: The generated character’s gender does not need to match the video content. It must strictly follow the gender specified in the 'gender' field, without exception. Additionally, use the 'character_depiction' section to identify traits corresponding to that gender when generating the image."
    else:
        Gender = "You don’t need to pay attention to the character’s gender in 'Gender'; it should match the video content."
    picture_prompt = f"""
    You are a professional MV visual director and art designer. Based on the following input, generate a concise and stylistically consistent JSON description.
    You must output the description in JSON format, following this data structure:
    {{
    "a_key_idea": str,        # concise summary of the scene’s central idea (≤ 5 words)
    "a_set_design": str,      # one-sentence visual description of the environment and tone
    "a_image_prompt": str     # prompt directly usable for AI image generation, must include the character’s visual traits (hair color, length, eye color, etc.) and maintain consistency of appearance
    }}
    # INPUT DATA
    This is the video content. Your task is to generate the first frame of the video:
    {para['story']}
    This is the image style. It should match the scene as closely as possible, but must not contain violence, pornography, gore, or politically sensitive elements. Keep it understated and subtle.
    {Gender}
    {qwen_result[para["number"]]}
    This is the character description. You must adhere to it to maintain character consistency. Use the story’s 'character_depiction' section to locate and generate the corresponding character.
    {label}
    Note: The generated content must not contain violence, pornography, gore, or politically sensitive material. It should not be exaggerated. Keep it subtle and restrained, and the character’s eyes must not glow.
    ### Requirements
    1. **You must output JSON format only.** Do not include any additional explanations.
    2. **Content must not contain violence, pornography, gore, or politically sensitive material.**
    3. **Maintain a restrained and understated tone**, emphasizing symbolism and atmosphere rather than exaggerated actions.
    4. The character description must specify distinct traits (e.g., hair color, hair length) and nationality, and be stylistically consistent with the rest of the content.
    5. The generated character’s eyes must not glow.
    """

    print({qwen_result[para["number"]]})
    response = client_doubao.chat.completions.create(

        model="doubao-seed-1.6-250615",
        messages=[
            {"role": "user", "content": picture_prompt}
        ],
        thinking={
            "type": "enabled" 
        },
    )
    picture = response.choices[0].message.content
    cleaned_picture = picture.strip()
    cleaned_picture = re.sub(r'^```json\n|\n```$', '', cleaned_picture) 
    print(cleaned_picture)
    # picture
    image_prompts = []
    matches = re.findall(
        r'(?:image_prompt["\']?\s*[:=]\s*["\']?|"image_prompt"\s*:\s*")([^"\'\n]+)',
        cleaned_picture,
        re.IGNORECASE 
    )
    image_prompts = list(set(matches))
    print("提取的 image_prompts:", image_prompts)
    image_prompts.sort()

    for prompt_p in image_prompts:
        prompt = prompt_p
        save_folder = f"./result/{name}/picture/{para['number']}" 
        retry = 3
        best_dict = {
            "image_url": "",
            "score": 0
        }
        while(retry):
            retry -= 1
            is_ok,best_dict = generate_image_doubao(prompt,save_folder,best_dict=best_dict)
        #   if is_ok:
        #       break
            if best_dict["score"] == 5:
                break
        # 获取图片的 URL
        image_url = best_dict["image_url"]

        # 设置保存图片的文件夹路径
        save_folder = f"./result/{name}/picture/{para['number']}"  
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)  
        image_name = f"1.jpg"  
        image_path = os.path.join(save_folder, image_name)
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_path, 'wb') as file:
                file.write(response.content)
            print(f"图片已成功保存为：{image_path}")
        else:
            print(f"下载图片失败，状态码：{response.status_code}")
        # 无Gemini_veirfy
        # prompt = prompt_p
        # save_folder = f"/map-vepfs/nicolaus625/m2v/tangxiaoxuan/work/{name}/picture/{para['number']}"  # 指定文件夹名
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)       
        # image_name = f"{len(os.listdir(save_folder)) + 1}.jpg" 
        # imagesResponse = client_doubao.images.generate(
        #     model="doubao-seedream-4-0-250828",
        #     prompt=prompt_p,
        #     extra_body={"watermark": False,"size": "1920x1080"}
        # )
        # image_url = imagesResponse.data[0].url
        # image_path = os.path.join(save_folder, image_name)
        # response = requests.get(image_url)
        # if response.status_code == 200:
        #     with open(image_path, 'wb') as file:
        #         file.write(response.content)
        #     print(f"图片已成功保存为：{image_path}")
        # else:
        #     print(f"下载图片失败，状态码：{response.status_code}")
    # camera
    all_shots = []
    for idx, prompt_p in enumerate(image_prompts, start=1):
        prompt = f"""
        I am now designing an MV shot.
        You are a **top-tier MV director**, and your task is to design a **concise short video shot script** based on the **image description**, **MV scene description**, **character settings**, and **style requirements** that I provide.

        In the `prompt`, clearly include the following **seven key elements**:

        1. **Subject** — The main focus of the video: person, animal, object, or scene (e.g., city, forest, dog).
        2. **Context** — The environment or setting where the subject is located (indoors, city street, forest, etc.).
        3. **Action** — What the subject is doing (walking, jumping, turning their head, etc.).
        4. **Style** — The cinematic or visual style (e.g., sci-fi, horror, cel-shaded animation).
        5. **Camera Motion & Angle** *(optional)* — e.g., aerial shot, eye level, top-down, low angle.
        6. **Composition** *(optional)* — The framing type, such as wide shot, close-up, single-person shot.
        7. **Ambiance** *(optional)* — Lighting and tone, such as warm tones, night scene, teal-blue color palette.

        Where:

        * `start` = {para['start']}
        * `end` = {para['end']}
        * `duration` = end - start
        * `shot_num` = 1
        * `label` = {para['label']}

        Please write everything **in English**.

        ---

        ### INPUT DATA

        * **Image description:** {prompt_p}
        * **MV scene description:** {para['story']} — primarily expand this into simple actions and camera movements.
        * **Character settings:** {label}

        ---

        You must output the result **in JSON format only**, without any extra explanation.
        Each JSON object should have the following structure:
        ```json
        {{
            "shot_num": int,          // Shot number
            "prompt": str,            // Shot description including the seven key elements, concise
            "camera_movement": str,   // Camera movement type, e.g. dolly, pan, tilt, etc.
            "shot_duration": float,   // Duration of the shot in seconds, designed according to lyric timestamps
            "start": float,           // Shot start time in seconds, based on lyric timestamps
            "end": float,             // Shot end time in seconds, based on lyric timestamps
            "label": str              // The lyric type corresponding to the shot, e.g., "story", "sing", etc.
        }}
        Only output one single shot (shot_num = 1) that best represents this scene.
        """
        response = client_doubao.chat.completions.create(
            model="doubao-seed-1.6-250615",
            messages=[
                {"role": "user", "content": prompt}
            ],
            thinking={
                "type": "enabled" 
        },
        )
        model_response= response.choices[0].message.content
        cleaned_text = model_response.strip()
        if cleaned_text.startswith('```json') and cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[7:-3].strip() 
        try:
            json_data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            raise ValueError(f"模型输出不是有效的 JSON:\n{cleaned_text}")
        all_shots.append(json_data) 
    save_folder = f"./result/{name}/camera"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    output_name = f"{para['number']}.json"
    output_path = os.path.join(save_folder, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_shots, f, ensure_ascii=False, indent=2) 

# begin
def generate_one_mv(name ,start_num=1,end_num=100,try_number= 1):
    audio_path_vocal = f"./result/{name}/{name}.mp3"
    audio_path = f"./result/{name}/{name}.mp3"
    lyrics = transcribe_audio(audio_path_vocal)
    songformer = read_json_file(f"./result/{name}/{name}.json")
    prompt = f"""
    1. **Lyrics with timestamps** (this is the *smallest lyric unit*):
    `{lyrics}`
    2. **Recognized song structure segments** (e.g. *intro / verse / chorus / bridge / outro*):
    `{songformer}`
    ---
    ### **Your Task**
    Based on the lyric timestamps and structural segmentation, generate a **complete MV storyboard time breakdown table**, following the specifications below:
    ---

    1. Time Segmentation Rules

    - The timestamps must continuously cover the entire song from start to finish.
    - Do not arbitrarily split lyric lines — each lyric line should remain intact whenever possible.
    - If a lyric segment is too short, you may merge it with an adjacent lyric, but after merging:
        - Each segment should have a duration between 3–8 seconds.
        - If merging causes the duration to exceed 8 seconds, keep them separate — do not force merging.
        - If a lyric’s original duration is too long, it may remain as is.
    - Strict rule: No segment may ever exceed 15 seconds in duration.
    - If any lyric or merged segment exceeds 15 seconds, you must forcibly split it into consecutive smaller segments.
    - You may split within a lyric line if necessary, but the split must follow natural pauses, rhythm, or phrasing.
    - **Important:** All `start` and `end` timestamps must align with 24fps time increments, rounded to two decimal places. Only use these exact increments:
    `[0.0, 0.04, 0.08, 0.12, 0.17, 0.21, 0.25, 0.29, 0.33, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.62, 0.67, 0.71, 0.75, 0.79, 0.83, 0.88, 0.92, 0.96]` per second fraction.
    - After calculating timestamps, round them to the nearest value from the list above.
    - The ending time should be based on the song structure segments, as the lyric timestamps are sometimes inaccurate.
    ---

    ### **2. Output Format**

    Output a **JSON array**, where each element represents an MV segment.
    The structure should be:

    ```json
    {{
    "number": 1,
    "start": 0.0,
    "end": 5.8,
    "label": "sing" or "story",
    "text": "lyric text",
    "story": "English description of the visual scene"
    }}
    ```

    * `number`: Segment index (starts from 1, increments sequentially)
    * `start` / `end`: Start and end timestamps (floating-point, in seconds)
    * `label`:
    * `"sing"` — shows the singer lip-syncing (no multiple people in frame)
    * `"story"` — represents narrative or visual storytelling scenes
    * `text`: The lyrics corresponding to this segment
    * `story`: The visual scene or storyboard description (**in English**) — should include visuals, character actions, lighting, and atmosphere
    ---
    ### **3. Visual & Narrative Logic**

    * Try to **identify the likely song** based on the lyrics, to infer the singer’s gender — ensure the `"sing"` segments reflect that gender.

    * Create a **cohesive MV narrative** (can be story-driven, emotional, or symbolic) that spans the entire song.

    * Alternate between `"sing"` and `"story"` segments logically:

    * Use a `"sing"` shot for the **first lyric line** in the *intro* or when transitioning between *verse* and *chorus*
    * Use a `"sing"` shot for the **first lyric** of each major section (e.g. verse, chorus)
    * Other parts can remain as `"story"` for narrative flow
    * You may occasionally integrate `"sing"` shots naturally within story sequences for variety

    * Each `"story"` field should be cinematic and descriptive, for example:

    * Lighting / color tone — e.g. `"warm amber light"`, `"neon reflection"`, `"cold blue twilight"`
    * Camera framing — e.g. `"Close-up on hands"`, `"Wide shot of city street"`
    * Action or mood — e.g. `"He looks away, lost in thought."`, `"Rain falls softly on his shoulders."`
    * For long segments, vary shots or describe minor visual transitions to avoid monotony.
    ---

    ### **4. Output Requirements**

    * Enclose the entire output within a code block:

    ```json
    [ ...your output here... ]
    ```
    * Ensure timestamps are **continuous and non-overlapping**.
    * The visuals must **match the song’s emotional tone**, forming a coherent MV narrative.

    """
    save_path = f"./result/{name}/story.json"

    if os.path.isfile(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            story = json.load(f)
    else:
        response_json = chat_with_gemini(prompt)
        story = extract_json_from_response(response_json)
        # 写入文件
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(story, f, ensure_ascii=False, indent=4)

    
    commands = generate_ffmpeg_commands(audio_path, story, name)

    for cmd in commands:
        os.system(cmd["command"])

    example="""
    {
      "good_designs": [
        "minimalist, sterile sci-fi aesthetic",
        "surreal, psychological horror elements (creeping ink, morphing reflections)",
        "high-contrast, expressionistic lighting",
        "use of digital glitches, distortion, and projection mapping"
      ],
      "style_requirement": "A cold, clinical, and minimalist aesthetic dominated by white, black, and cold blue light. The palette should be stark and desaturated, with warmth only appearing in deceptive, glitching flashbacks. Lighting transitions from sterile and flat to dynamic, high-contrast, and chaotic, mirroring the character's psychological breakdown and transformation. The overall atmosphere is tense, sterile, and surreal, evolving into something cathartic and empowering.",
      "user_region": "Western",
    {
      "good_designs": [
        "minimalist, sterile sci-fi aesthetic",
        "surreal, psychological horror elements (creeping ink, morphing reflections)",
        "high-contrast, expressionistic lighting",
        "use of digital glitches, distortion, and projection mapping"
      ],
      "style_requirement": "A cold, clinical, and minimalist aesthetic dominated by white, black, and cold blue light. The palette should be stark and desaturated, with warmth only appearing in deceptive, glitching flashbacks. Lighting transitions from sterile and flat to dynamic, high-contrast, and chaotic, mirroring the character's psychological breakdown and transformation. The overall atmosphere is tense, sterile, and surreal, evolving into something cathartic and empowering.",
      "user_region": "International / Western",
      "character_depiction": {
        "The Subject": {
          "name": "The Subject",
          "age": "late 20s",
          "gender": "male",
          "nationality": "US"
          "hair_color": "dark brown",
          "hair_length": "short, slightly unkempt",
          "appearance": "a haunted, introspective man trapped between guilt and awakening, his expression shifting from numbness to defiant resolve",
          "role": "protagonist / psychological prisoner"
        },
        "The Manipulator": {
          "name": "The Manipulator",
          "age": "late 20s",
          "gender": "female",
          "nationality": "US"
          "hair_color": "light brown in memory, dark brown in reality",
          "hair_length": "medium, softly flowing then disheveled",
          "appearance": "a shifting figure—warm and tender in illusions, cold and distorted in truth, embodying control and deceit",
          "role": "antagonist / embodiment of control"
        }
      }
    }
    """

    prompt = f"""
      story: {story}
      You are now a professional MV script master. Please create character portraits for the main characters in this story, using the names that appear in the script as their identifiers.
      You must explicitly include attributes such as hair color and hair length in the fields.
      Fill in each field of the JSON structure based on the story content, and output in JSON format, wrapping the result inside json code fences.
      Pay attention, the styles in good_designs should not have coexistence of anime and realism or similar situations. That is, regardless of the style, it's best to maintain consistency.
      {example}
    """
    save_path = f"./result/{name}/label.json"

    if os.path.isfile(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            label = json.load(f)
    else:
        response_json = chat_with_gemini(prompt)
        label = extract_json_from_response(response_json)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=4)
    audio_dir = f"./result/{name}/piece"
    audio_files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")])
    qwen_result = qwen_api(audio_files)
    # 选择需要生成的段落
    selected_paras = [p for p in story if start_num <= p['number'] <= end_num]

    # 并发执行 generate
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(generate, para, label, qwen_result, name, try_number) for para in selected_paras]
        for future in as_completed(futures):
            try:
                future.result() 
            except Exception as e:
                print("⚠️ 某个任务失败：", e)
