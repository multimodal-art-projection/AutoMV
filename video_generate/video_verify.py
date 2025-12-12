import base64
import os

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
import ffmpeg
from config import Config

def parse_json_output(output):
    match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)

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
template = '''
Please judge whether the video content meets the requirements. The requirements are as follows:
1. Does the video content or instruction contain anything that violates physics?
2. Does the video content generally follow the scene content in the instruction without any sense of inconsistency? However, you don't need to consider whether the camera movement is followed.

You also need to rate the quality of the video content, giving low scores to videos with content anomalies (abnormal character actions). The maximum score is 5 points, and the scoring criteria are as follows:
**1 point**: The video content completely fails to meet the requirements, with serious violations of physics, or the content is completely inconsistent with the instruction requirements. Character actions are unnatural, choppy, or unreasonable, the picture quality is poor, and it affects audience understanding.

**2 points**: The video content basically meets the requirements, but there are obvious violations of physics, or some scenes fail to follow the instruction content. Character actions are somewhat uncoordinated or not smooth, some scenes have poor quality, causing discomfort when viewing.

**3 points**: The video content generally meets the requirements, but occasionally has minor violations of physics, and some scenes have slight deviations from the instruction content. Character actions are basically natural, but some parts don't look realistic enough or appear slightly stiff. Picture quality is average, and overall it can be understood and accepted.

**4 points**: The video content meets the requirements well, with very few violations of physics or minor inconsistencies. Character actions are smooth and natural, and the picture quality is good. Overall it follows the instruction content, with basically no sense of inconsistency when viewing, and good visual presentation.

**5 points**: The video content completely meets the requirements, with no violations of physics, and completely follows the scene content in the instruction. Character actions are natural and smooth, the picture quality is excellent, the visual effect is highly consistent with the instruction requirements, and the viewing experience is perfect.

Instruction: [INSTRUCTION]
If it meets the requirements, please output yes, otherwise output no. If the instruction itself would lead to results that violate physics, please output the reason at the end.
Output in JSON format:
```json
{
    "judge result": <yes or no>,
    "score": ...,
    "reason": ...
}```
'''
def call_gemini(instruction, video_file_path, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try: 
            client = OpenAI(
                api_key=Config.GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            with open(video_file_path, "rb") as f:
                video_bytes = base64.b64encode(f.read()).decode('utf-8')
            prompt = template.replace("[INSTRUCTION]",instruction)
            response = client.chat.completions.create(
                model= "gemini-2.5-flash", # 'gemini-2.5-pro',
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt
                        },
                        {
                            "type": "video_url", 
                            "video_url": {
                                "url": f"data:video/mp4;base64,{video_bytes}"
                            }
                        }
                    ]
                }]
            )
            print("Gemini API Response:", response.choices[0].message.content)
            result = parse_json_output(response.choices[0].message.content)
            if result and type(result) == dict:
                return result
            else:
                return call_gemini(instruction, video_file_path)

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