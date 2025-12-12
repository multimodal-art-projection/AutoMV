import base64
import os
import concurrent.futures
import requests
from google import genai
from google.genai import types
import base64
import time
from google import genai
from google.genai import types
import re
import json
import tqdm
from openai import OpenAI
from config import Config
def parse_json_output(output):
    match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
    # print("match:", match)
    if match:
        json_str = match.group(1)
        json_str = json_str.replace('\n', '').replace('\r', '').strip()
        try:
            json_dict = json.loads(json_str)
            return json_dict
        except json.JSONDecodeError as e:
            pass
    else:
        pass
    return None
Prompt_Template = """
Here is a segment of shot content and its camera movement description. Please break this shot into different segments in chronological order, each lasting 8 seconds. If a segment is shorter than 8 seconds, describe the remaining content. For example, if a segment is 18.8 seconds long, you should output three segments: the first one for 0.00-8.00s with its shot content and corresponding camera movement, the second for 8.00-16.00s, and the third for 16.00-18.80s.

Complete shot content and camera movement description as follows: [DESCRIPTION]
Shot duration: [TIME]

Output format requirements:
1. The result must be a valid JSON array.
2. The entire JSON output must be wrapped with triple backticks.
3. The opening must be ```json and the closing must be ``` exactly. Do not omit the final backticks.
4. Camera movement descriptions should remain accurate and cinematic.
5. Ensure that the sum of the shot durations in the storyboard matches the shot duration before splitting.
Example:
```json
[{
  "shot_num": 1,
  "prompt": "",
  "camera_movement": "",
  "shot_duration": 
},
{
  "shot_num": 2,
  "prompt": "",
  "camera_movement": "",
  "shot_duration": 
}]
```

Output JSON format:
"""
def call_gemini(prompt, max_retries=5):
    retry_count = 0
    while retry_count < max_retries:
        # try:
        client = OpenAI(
        api_key=Config.DOUBAO_API_KEY,
        base_url="https://ark.cn-beijing.volces.com/api/v3"
        )

        response = client.chat.completions.create(
            model="doubao-seed-1.6-250615",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    
        return response.choices[0].message.content



def generate_shots(json_file):
    print("JSON FILE:", json_file)
    with open(json_file, 'r') as f:
        data = json.load(f)[0]
    instruction = data['prompt']
    camera_movement = data['camera_movement']
    shot_time = str(data['shot_duration'])
    description = f'''
    Instruction: {instruction}
    Camera movement: {camera_movement}
    '''
    prompt = Prompt_Template.replace("[DESCRIPTION]",description).replace("[TIME]",shot_time)
    output = call_gemini(prompt)
    if output[-3:] != "```":
        result = parse_json_output(output+"```")
    else:
        result = parse_json_output(output)
    print("result:", result)
    if result:
        return result
    else:
        print("retry")
        return generate_shots(json_file)


def process_json_file(file_folder,json_file):
    json_file_path = os.path.join(file_folder, json_file)
    output_file_path = json_file_path[:-5] + "_1.json"
    if os.path.exists(output_file_path):
        print(f"Output file {output_file_path} already exists. Skipping...")
        return None
    with open(json_file_path, 'r') as f:
        data = json.load(f)[0]
        label = data.get('label', None)
        if label is None:
            print(f"No label found in {json_file}. Skipping...")
            return None
        if label == "sing":
            print(f"It's a singing scene in {json_file}. Skipping...")
            return None
    
    # Assuming `generate_shots` is a function you've defined elsewhere.
    shots = generate_shots(json_file_path)
    with open(output_file_path, 'w') as f:
        json.dump(shots, f)
    
    return json_file

def cut_camera(file_folder):
    json_files = [
        f for f in os.listdir(file_folder)
        if f.endswith('.json')
        and not f.endswith('_1.json')
        and f"{os.path.splitext(f)[0]}_1.json" not in os.listdir(file_folder)
    ]
    # print(json_files)
    # exit()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks for each JSON file to the thread pool
        futures = {executor.submit(process_json_file, file_folder, json_file): json_file for json_file in json_files}
        
        # Wait for results
        for future in concurrent.futures.as_completed(futures):
            json_file = futures[future]
            # try:
            result = future.result()
            if result:
                print(f"Successfully processed {json_file}")

if __name__ == "__main__":
    file_folder = "/map-vepfs/nicolaus625/m2v/tangxiaoxuan/work/music_pipeline/15/camera1"
    cut_camera(file_folder)