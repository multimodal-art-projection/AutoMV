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
def parse_json_output(output):
    match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)

    if match:
        json_str = match.group(1)
        # Remove newlines and extra spaces
        json_str = json_str.replace('\n', '').replace('\r', '').strip()
        try:
            json_dict = json.loads(json_str)
            return json_dict
        except json.JSONDecodeError as e:
            pass
    else:
        pass
    return None

technical_template = '''
You are an MV appreciation expert. You need to score the MV according to the evaluation criteria and provide analysis.
I. Scoring domain: Technical Module
Scoring criteria:
- Character Consistency 5 points:
    1: Character image changes frequently, facial features, clothing, body shape, etc. show obvious inconsistencies, viewers cannot confirm it is the same character
    2: Character image has more than 3 obvious inconsistencies, such as sudden changes in facial features, hairstyle, etc. affecting viewing experience
    3: Character image is basically consistent, with 1-2 minor inconsistencies that do not affect overall viewing
    4: Character image remains highly consistent, details (such as makeup, accessories) transition naturally between scenes
    5: Character image is perfectly consistent throughout, even under complex lighting and pose changes, details remain precise
- Physical Realism 5 points:
    1: Physical rules are seriously violated throughout, movements are stiff, object/character motion violates basic physics, obviously illogical
    2: Multiple (more than 5) physical inconsistencies, such as floating, clipping, incorrect object interactions, discontinuous movements, abnormal gravity, etc.
    3: Basically follows physical rules, simple movements are natural, but complex interactions (multi-object collision, fluid, cloth, etc.) have obvious flaws
    4: Physical effects are close to reality, interactions (character actions, environment elements, special effects) basically follow physical rules, minor detail issues
    5: Perfect physical performance, all element interactions are realistic, including complex actions, environmental physics (water, fire, smoke), tiny details follow natural laws
- Lip Sync Accuracy 5 points:
    1: Lip movements completely do not match lyrics throughout, obvious misalignment or static mouth
    2: More than 30% of lyric segments are not matched, high notes/special pronunciations are obviously wrong
    3: Main lyrics are synced, but details (such as consonants, long sounds) are not precise, about 10-20% not synced
    4: More than 95% of lip sync is perfect, including fast-paced segments, only a few complex pronunciations slightly off
    5: Professional-level lip sync, including all syllables, breaths, emotional changes, indistinguishable from live performance by professional singer
- Visual Style Harmony 5 points:
    1: Visual style is chaotic, color tone, texture, rendering style between scenes are completely inconsistent
    2: Obvious style breaks, such as realistic scene suddenly switching to cartoon style without transition
    3: Basic style unity, some scenes (such as effects, dream sequences) have style differences but with clear intent
    4: Highly unified visual style, scene changes maintain consistent aesthetics, color schemes are coordinated
    5: Perfect visual consistency, rich visual layers within unified style, forming a unique visual identity

II. Scoring domain: Post-production Editing
Scoring criteria:
- Shot Continuity 5 points:
    1: Shot transitions are stiff, many jump cuts, spatial/temporal logic is chaotic, viewers cannot follow
    2: Basic editing skills lacking, transitions are crude, rhythm missing, many scenes are not continuous
    3: Uses conventional editing techniques, shot transitions are basically smooth, rhythm matches music roughly
    4: Skilled editing, rich shot language, creative transitions, clear spatial logic, precise rhythm control
    5: Master-level editing, perfect shot narrative, every cut has artistic purpose, forms unique editing style
- Audio-Visual Correlation 5 points:
    1: Visuals and music are almost unrelated, rhythm, emotion, key segments are not synchronized
    2: Basic audio-visual sync (e.g. drum beat matches cut), but lacks deeper correlation
    3: Important music nodes (chorus, climax) have clear visual response, basic emotion matches
    4: Precise audio-visual sync, including rhythm, emotional layers, musical details all have visual correspondence
    5: Music and visuals achieve symbiosis, visual elements become an extension of music, forming unique audiovisual language

III. Scoring domain: Content Module
Scoring criteria:
- Music Theme Fit 5 points:
    1: Video content is completely unrelated to song theme, music emotion and visual emotion contradict each other
    2: Surface fit (such as literal lyrics), but lacks understanding and expression of deeper musical meaning
    3: Accurately grasps core music theme, visual narrative basically matches song emotion
    4: Deeply explores song connotation, enriches musical expression through visual metaphor/symbolism
    5: Video becomes an inseparable part of music, expands musical meaning from unique perspective, mutually enhances artistic height
- Storytelling 5 points:
    1: No clear narrative structure, random scene splicing, viewers cannot understand story/theme
    2: Basic narrative elements but chaotic logic, unclear character motivation, plot development lacks continuity
    3: Complete narrative structure (beginning-development-climax-ending), basic plot logic is clear
    4: Well-designed narrative structure, rich character portrayal, meaningful plot twists, clear theme
    5: Innovative narrative techniques, multi-layered story structure, leaves room for thought, maintains emotional resonance, reaches short film-level storytelling
- Emotional Expressiveness 5 points:
    1: Emotional expression is bland/false, character expressions/actions lack emotional persuasiveness, viewers cannot engage
    2: Emotional expression is single-layered, lacks variation, fails to touch viewers' emotional resonance
    3: Basic emotional expression in place, character emotion changes naturally, can elicit basic empathy
    4: Rich multi-layered emotional expression, subtle emotional changes, can trigger deep emotional resonance
    5: Emotional expression reaches artistic sublimation, creates profound emotional experience through exquisite audiovisual language, produces lasting emotional impact

IV. Scoring domain: Artistic Module
Scoring criteria:
- Visual Composition and Texture 5 points:
    1: Poor image quality, low resolution, random composition, lack of lighting, texture like early amateur productions
    2: Basic image clarity, but bland composition, simple lighting, texture like ordinary online video
    3: Professional image quality, standard composition, basic lighting design, texture close to commercial MV standard
    4: High-quality visual presentation, carefully designed composition and lighting, harmonious color aesthetics, excellent texture
    5: Cinematic visual quality, every frame is meticulously constructed, lighting and color reach artistic level, can be appreciated as visual art alone
- Creative Uniqueness 5 points:
    1: No innovation throughout, completely copies existing MV templates, concept highly similar to works in past three years
    2: Only 1-2 common creative points (such as conventional transitions/basic effects), core concept lacks uniqueness
    3: Clear theme innovation (such as narrative structure reorganization), but execution refers to existing cases
    4: Breakthrough visual symbols (such as new camera device), at least 3 scenes realize conceptual innovation
    5: Original worldview throughout, at least two scenes/camera movements subvert traditional MV design paradigms
AI Feature Display 5 points:
    1: No use of AI features at all, or deliberately mimics traditional shooting to cover up AI traits
    2: AI traits passively appear (such as occasional deformation/style breaks), but not integrated into creative design
    3: Conscious display of AI aesthetics (such as style fusion, surreal deformation), but stays at technical demonstration level
    4: Creatively uses AI features as means of expression, forms unique visual language, serves narrative/emotion
    5: Elevates AI features to artistic language, creates visual "wonders" impossible for traditional photography, while maintaining artistic integrity

Please return the scoring result in JSON format
```json{
    "Technical Module":{
        "Character Consistency": <score>,
        "Physical Realism": <score>,
        "Lip Sync Accuracy": <score>,
        "Visual Style Harmony": <score>
    },
    "Post-production Editing":{
        "Shot Continuity": <score>,
        "Audio-Visual Correlation": <score>
    },
    "Content Module":{
        "Music Theme Fit": <score>,
        "Storytelling": <score>,
        "Emotional Expressiveness": <score>
    },
    "Artistic Module":{
        "Visual Composition and Texture": <score>,
        "Creative Uniqueness": <score>,
        "AI Feature Display": <score>
}
```
'''
shot_template = '''
You are an MV appreciation expert. You need to score the MV according to the evaluation criteria and provide analysis. Scoring domain: Post-production Editing
Scoring criteria:
- Shot Continuity 5 points:
    1: Shot transitions are stiff, many jump cuts, spatial/temporal logic is chaotic, viewers cannot follow
    2: Basic editing skills lacking, transitions are crude, rhythm missing, many scenes are not continuous
    3: Uses conventional editing techniques, shot transitions are basically smooth, rhythm matches music roughly
    4: Skilled editing, rich shot language, creative transitions, clear spatial logic, precise rhythm control
    5: Master-level editing, perfect shot narrative, every cut has artistic purpose, forms unique editing style
- Audio-Visual Correlation 5 points:
    1: Visuals and music are almost unrelated, rhythm, emotion, key segments are not synchronized
    2: Basic audio-visual sync (e.g. drum beat matches cut), but lacks deeper correlation
    3: Important music nodes (chorus, climax) have clear visual response, basic emotion matches
    4: Precise audio-visual sync, including rhythm, emotional layers, musical details all have visual correspondence
    5: Music and visuals achieve symbiosis, visual elements become an extension of music, forming unique audiovisual language

Please return the scoring result in JSON format
```json{
    "Shot Continuity": <score>,
    "Audio-Visual Correlation": <score>
}
```
'''
content_template = '''
You are an MV appreciation expert. You need to score the MV according to the evaluation criteria and provide analysis. Scoring domain: Content Module
Scoring criteria:
- Music Theme Fit 5 points:
    1: Video content is completely unrelated to song theme, music emotion and visual emotion contradict each other
    2: Surface fit (such as literal lyrics), but lacks understanding and expression of deeper musical meaning
    3: Accurately grasps core music theme, visual narrative basically matches song emotion
    4: Deeply explores song connotation, enriches musical expression through visual metaphor/symbolism
    5: Video becomes an inseparable part of music, expands musical meaning from unique perspective, mutually enhances artistic height
- Storytelling 5 points:
    1: No clear narrative structure, random scene splicing, viewers cannot understand story/theme
    2: Basic narrative elements but chaotic logic, unclear character motivation, plot development lacks continuity
    3: Complete narrative structure (beginning-development-climax-ending), basic plot logic is clear
    4: Well-designed narrative structure, rich character portrayal, meaningful plot twists, clear theme
    5: Innovative narrative techniques, multi-layered story structure, leaves room for thought, maintains emotional resonance, reaches short film-level storytelling
- Emotional Expressiveness 5 points:
    1: Emotional expression is bland/false, character expressions/actions lack emotional persuasiveness, viewers cannot engage
    2: Emotional expression is single-layered, lacks variation, fails to touch viewers' emotional resonance
    3: Basic emotional expression in place, character emotion changes naturally, can elicit basic empathy
    4: Rich multi-layered emotional expression, subtle emotional changes, can trigger deep emotional resonance
    5: Emotional expression reaches artistic sublimation, creates profound emotional experience through exquisite audiovisual language, produces lasting emotional impact

Please return the scoring result in JSON format
```json{
    "Music Theme Fit": <score>,
    "Storytelling": <score>,
    "Emotional Expressiveness": <score>
}
```
'''
art_template = '''
You are an MV appreciation expert. You need to score the MV according to the evaluation criteria and provide analysis. Scoring domain: Artistic Module
Scoring criteria:
- Visual Composition and Texture 5 points:
    1: Poor image quality, low resolution, random composition, lack of lighting, texture like early amateur productions
    2: Basic image clarity, but bland composition, simple lighting, texture like ordinary online video
    3: Professional image quality, standard composition, basic lighting design, texture close to commercial MV standard
    4: High-quality visual presentation, carefully designed composition and lighting, harmonious color aesthetics, excellent texture
    5: Cinematic visual quality, every frame is meticulously constructed, lighting and color reach artistic level, can be appreciated as visual art alone
- Creative Uniqueness 5 points:
    1: No innovation throughout, completely copies existing MV templates, concept highly similar to works in past three years
    2: Only 1-2 common creative points (such as conventional transitions/basic effects), core concept lacks uniqueness
    3: Clear theme innovation (such as narrative structure reorganization), but execution refers to existing cases
    4: Breakthrough visual symbols (such as new camera device), at least 3 scenes realize conceptual innovation
    5: Original worldview throughout, at least two scenes/camera movements subvert traditional MV design paradigms
AI Feature Display 5 points:
    1: No use of AI features at all, or deliberately mimics traditional shooting to cover up AI traits
    2: AI traits passively appear (such as occasional deformation/style breaks), but not integrated into creative design
    3: Conscious display of AI aesthetics (such as style fusion, surreal deformation), but stays at technical demonstration level
    4: Creatively uses AI features as means of expression, forms unique visual language, serves narrative/emotion
    5: Elevates AI features to artistic language, creates visual "wonders" impossible for traditional photography, while maintaining artistic integrity

Please return the scoring result in JSON format
```json{
    "Visual Composition and Texture": <score>,
    "Creative Uniqueness": <score>,
    "AI Feature Display": <score>
}
```
'''

area_template = {
    "Content Module": content_template,
    "Artistic Module": art_template,
    "Technical Module": technical_template,
    "Post-production Editing": shot_template,
}

def call_gemini(prompt, video_file_path, max_retry=10):
    """
    Call Gemini API to process video file

    Args:
        prompt: prompt text
        video_file_path: video file path
        max_retry: max retry count

    Returns:
        Parsed JSON result
    """

    for attempt in range(max_retry):
        try:
            video_bytes = open(video_file_path, 'rb').read()

            client = genai.Client(
                            api_key="AIzaSyBHBpSyAK6xBsw6J4-p-GLOonzVsO-FBEY"
                        )
            response = client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=video_bytes,
                                mime_type='video/mp4'),
                            video_metadata=types.VideoMetadata(fps=0.5,start_offset='0s',
                    end_offset='120s')
                        ),
                        types.Part(text=prompt)
                    ]
                )
            )
            result = response.text
            print("\n" + "="*50)
            print(result)
            print("="*50 + "\n")

            # Parse JSON
            json_result = parse_json_output(result)

            if json_result:
                print("✓ Successfully obtained valid JSON result")
                return json_result
            else:
                print(f"✗ JSON parsing failed, preparing to retry...")
                if attempt < max_retry - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    raise ValueError("Max retry count reached, still unable to obtain valid JSON")

        except ValueError as e:
            print(f"✗ Error: {e}")
            if attempt < max_retry - 1:
                print(f"Will retry in 2 seconds for attempt {attempt + 2}...")
                time.sleep(2)
            else:
                print(f"Max retry count reached ({max_retry}), operation failed")
                raise

        except Exception as e:
            error_message = str(e)
            print(f"✗ Error occurred: {type(e).__name__}: {e}")

            # Decide whether to reset file only for file-related errors
            should_reset_file = False

            # Check for file-related errors
            if any(keyword in error_message.lower() for keyword in ['file', 'upload', 'invalid', 'not found']):
                if 'overloaded' not in error_message.lower() and '503' not in error_message:
                    should_reset_file = True
                    print("File-related error detected, will re-upload file on next retry")

            if should_reset_file:
                myfile = None

            if attempt < max_retry - 1:
                # Adjust wait time based on error type
                if '503' in error_message or 'overloaded' in error_message.lower():
                    wait_time = min(5 * (attempt + 1), 10)  # Exponential backoff, max 30s
                    print(f"Model overloaded, will retry in {wait_time} seconds for attempt {attempt + 2}...")
                    time.sleep(wait_time)
                elif '429' in error_message or 'quota' in error_message.lower():
                    wait_time = min(10 * (attempt + 1), 60)  # Longer wait for quota errors
                    print(f"Quota limit, will retry in {wait_time} seconds for attempt {attempt + 2}...")
                    time.sleep(wait_time)
                else:
                    print(f"Will retry in 2 seconds for attempt {attempt + 2}...")
                    time.sleep(2)
            else:
                print(f"Max retry count reached ({max_retry}), operation failed")
                raise

    # Should not reach here, but for completeness
    raise RuntimeError("Retry logic terminated abnormally")


def llm_as_a_judge(video_file_path):
    prompt = area_template["Technical Module"]
    json_result = call_gemini(prompt, video_file_path)
    return json_result
def get_total_score(video_file_path):
    total_score = 0
    result = llm_as_a_judge(video_file_path)
    print("Evaluation result:",result)
    return total_score, result
if __name__ == '__main__':
    # get_total_score('/Users/leixinping/code/4 (online-video-cutter.com).mp4')
    video_folder_path = ''
    record_file = ''
    with open(record_file, 'a') as f:
            f.write('')
    # Get completed video list
    completed_videos = []
    with open(record_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            completed_videos.append(record['video_file'])
    video_files = os.listdir(video_folder_path)
    video_files = [video_file for video_file in video_files if (video_file not in completed_videos) and video_file.endswith('.mp4')]
    for video_file in video_files:
        record = {}
        record["video_file"] = video_file
        print("Processing:",video_file)
        video_file_path = os.path.join(video_folder_path, video_file)
        _, result_dict = get_total_score(video_file_path)
        record["result"] = result_dict
        with open(record_file, 'a') as f:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
            f.write('\n')
