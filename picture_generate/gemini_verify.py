import base64
import os

import requests
import base64
import time
import re
import json
import tqdm
from openai import OpenAI
import base64
from pathlib import Path
import requests
import json
from config import Config

def parse_json_output(output):
    match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)

    if match:
        json_str = match.group(1)
        # 去掉换行符和多余的空格
        json_str = json_str.replace('\n', '').replace('\r', '').strip()
        try:
            json_dict = json.loads(json_str)
            return json_dict
        except json.JSONDecodeError as e:
            pass
    else:
        pass
    return None
template = '''
请你判断图像内容是否符合要求，要求如下：
1. 是否出现反物理现象，是否为正常的图片？例如：人物的身体部位是否符合正常认知，图像中各部分的相对大小是否正确？注意图片不要是动画片风格。
2. 是否大致遵循了instruction
Instruction:[INSTRUCTION]
如果符合要求,请你输出yes,否则输出no.
此外，你需要对图片与指令的匹配程度以及图像内容质量，请给出一个分数，分数越高，匹配程度越高，满分5分。
评分标准为：
1分：完全不相关：图片内容与指令无关或出现不正常的画面,图片为动画片风格。
2分：部分相关：图片内容与指令有一定的匹配，但并不完整。
3分：大部分匹配：图片内容与指令大部分匹配，但可能有些小的偏差。
4分：基本匹配：图片内容与指令几乎完全匹配，仅有微小差异。
5分：完全匹配：图片内容与指令完全一致。

请以JSON格式输出:
```json
{
    "judge result":<yes or no>,
    "score": <score>,
    "reason": ...
}
```
'''
def call_gemini(instruction, image_path, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            client = OpenAI(
                api_key=Config.GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            with open(image_path, "rb") as f:
                image_bytes = base64.b64encode(f.read()).decode('utf-8')
            prompt = template.replace("[INSTRUCTION]",instruction)
            response = client.chat.completions.create(
                model='gemini-2.5-pro',
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_bytes}"
                            }
                        }
                    ]
                }]
            )
            print("Gemini API Response:", response.choices[0].message.content)
            result = parse_json_output(response.choices[0].message.content)
            if result:
                return result
            else:
                return call_gemini(instruction, image_path)

        except requests.exceptions.RequestException as req_error:
            print(f"Error: A request error occurred while calling the Gemini API. {req_error}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying... Attempt {retry_count}/{max_retries}")
                time.sleep(5)
            else:
                print(f"Failed after {max_retries} retries.")
                return f"Error: A request error occurred while calling the Gemini API. {req_error}"

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying... Attempt {retry_count}/{max_retries}")
                time.sleep(5)
            else:
                print(f"Failed after {max_retries} retries.")
                return f"An unexpected error occurred: {e}"

    return "Error: Max retry attempts reached. Unable to complete the request."



if __name__ == '__main__':
    picture_path = '/map-vepfs/nicolaus625/m2v/leixinping/FlexWorld/video/ark-image-generate.jpeg'
    instruction = '''A close-up captures hands gently turning a vase upsi
de down. With precise, intentional movements, they inscribe a maker's mark in el
egant Lishu script onto its base. The scene is set in a minimalist room, bathed 
in moonlight from a floor-to-ceiling window that reveals distant city lights. Th
e aesthetic is photorealistic with a wuxia film influence, using a palette of bl
ue-white and muted earth tones to create a serene and introspective ambiance.", 
"camera_movement": "Static close-up with a subtle focus pull to the writing hand.'''
    result = call_gemini(instruction, picture_path)
    print(result)