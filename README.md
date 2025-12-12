# AutoMV: Automatic Multi-Agent System for Music Video Generation

AutoMV is a **training-free, multi-agent system** that automatically generates coherent, long-form **music videos (MVs)** directly from a full-length song.  
The pipeline integrates music signal analysis, scriptwriting, character management, adaptive video generation, and multimodal verification—aiming to make high-quality MV production accessible and scalable.

This repository corresponds to the paper:

> **AutoMV: An Automatic Multi-Agent System for Music Video Generation**

---

## 🚀 Features

AutoMV is designed as a full music-to-video (M2V) production workflow with strong music-aware reasoning abilities.

### 🎼 Music Understanding and Preprocessing

- Beat tracking, structure segmentation (**SongFormer**)
- Vocal/accompaniment separation (**htdemucs**)
- Automatic lyrics transcription with timestamps (**Whisper**)
- Music captioning (genre, mood, vocalist attributes) using **Qwen2.5-Omni**

### 🎬 Multi-Agent Pipeline

- **Screenwriter Agent**: creates narrative descriptions, scene summaries, character settings  
- **Director Agent**: produces shot-level scripts, camera instructions, and prompts  
- **Verifier Agent**: checks physical realism, instruction following, and character consistency

### 🧍 Character Bank

- A structured database describing each character’s:  
  *face*, *hair*, *skin tone*, *clothing*, *gender*, *age*, etc.
- Ensures stable identity across multiple shots and scenes

### 🎥 Adaptive Video Generation Backends

- **Doubao Video API**: general cinematic shots  
- **Qwen-Wan 2.2**: lip-sync shots using vocal stems  
- Keyframe-guided generation with cross-shot continuity

### 🧪 Full-Song MV Benchmark (First of Its Kind)

Includes **12 fine-grained criteria** under 4 categories:

- Technical
- Post-production
- Content
- Art

Evaluated via **LLM judges** (Gemini-2.5-Pro/Flash) and **human experts**.

---

## 🧩 System Overview

AutoMV consists of four main stages:

1. **Music Preprocessing**  
2. **Screenwriter & Director Agents**  
3. **Keyframe + Video Clip Generation**  
4. **Gemini Verifier & Final Assembly**

A detailed architecture diagram is available in the paper.

---

## 📦 Installation

AutoMV is a training-free system, relying on MIR tools and LLM/VLM APIs.

### 1. Clone the repository

```bash
git clone https://github.com/xxx/AutoMV.git
cd AutoMV
```

### 2. Install dependencies

```bash
pip install -r SongFormer_requirements.txt
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

Dependencies include:

- `ffmpeg`
- `htdemucs`
- `whisper`
- `pydub`
- SDKs for Gemini, Doubao, Qwen, etc.


This information has been organized and translated into English Markdown format:

### 3\. Add Environment Variables

Export the following Environment Variables in your shell profile (e.g., `.bashrc`, `.zshrc`) or set them as environment variables before running the project:

```bash
GEMINI_API_KEY=xxx
DOUBAO_API_KEY=xxx
ALIYUN_OSS_ACCESS_KEY_ID=xxx  # Aliyun OSS Access Key ID
ALIYUN_OSS_ACCESS_KEY_SECRET=xxx  # Aliyun OSS Access Key Secret
ALIYUN_OSS_BUCKET_NAME=xxx  # Aliyun OSS Bucket Name
HUOSHAN_ACCESS_KEY=xxx  # Huoshan Engine ACCESS KEY
HUOSHAN_SECRET_KEY=xxx  # Huoshan Engine SECRET KEY
GPU_ID=xxx  # Optional
WHISPER_MODEL=xxx
QWEN_OMNI_MODEL=xxx
```

### 4\. Download Required Models

Before running the project, **download the following pretrained models**:

1.  **Qwen2.5-Omni-7B**

      * **Download Source:** ModelScope
      * **Link:** [https://modelscope.cn/models/qwen/Qwen2.5-Omni-7B](https://modelscope.cn/models/qwen/Qwen2.5-Omni-7B)

2.  **Whisper Large-v2**

      * **Installation & Usage Instructions:**
      * **Link:** [https://github.com/openai/whisper](https://github.com/openai/whisper)

3.  **Wan2.2-s2v (Optional)**

    **Note:** This model is for local lip-synced video generation. Processing a single song typically requires **4-5 hours on an A800 GPU**, but it is significantly cheaper than using API calls.

   * **Model Setup:**

      1.  Navigate to the lip-sync directory:
         ```cd generate_lip_video```
      2.  Clone the model repository:
         ```git clone https://huggingface.co/Wan-AI/Wan2.2-S2V-14B```
      3.  **Environment Setup (Mandatory due to conflicts):**
         A new environment is required for the local model due to potential package conflicts.
         ```bash
         conda create -n gen_lip python=3.10
         conda activate gen_lip
         pip install requirements.txt
         pip install requirements_s2v.txt
         ```
      4.  **Code Modification:**
         Comment out the function call `gen_lip_sync_video_jimeng(music_video_name, config = Config)` within the file `generate_pipeline.py`.

      * **Testing/Execution Steps (Once config setup is complete):**
      ```
      # 1. Navigate to the picture generation directory:
      cd picture_generate
      # 2. Run the picture generation script:
      python picture.py
      # 3. Run the lip-sync generation script:
      python generate_lip_video/gen_lip_sycn_video.py
      # 4. Run the main pipeline:
      python generate_pipeline.py
      ```

After downloading the models, specify their paths in `config.py`:

```bash
MODEL_PATH_QWEN = "/path/to/Qwen2.5-Omni-7B"
WHISPER_MODEL_PATH = "/path/to/whisper-large-v2"
```

#### SongFormer

Download Pre-trained Models

```bash
cd picture_generate/SongFormer/src/SongFormer
# For users in mainland China, you may need export HF_ENDPOINT=https://hf-mirror.com
python utils/fetch_pretrained.py
```

---

## 🎧 Usage

### 1. Prepare your audio

Place your `.mp3` or `.wav` file into:

```bash
./result/{music_name}/{music_name}.mp3
```

### 2. Run AutoMV

In `config.py`, replace `{music_name}` with the identifier of your music project.  
This name will be used as the directory name for storing all intermediate and final outputs.
Please use only English letters, numbers, or underscores in the name.

For users in mainland China, you may need export HF_ENDPOINT=https://hf-mirror.com

(1) Generate the first-frame images for each MV segment

```bash
python -m picture_generate.main
```

This step:

- Generates visual prompts for each segment
- Produces keyframe images
- Saves results under result/{music_name}/picture/

(2) Generate the complete music video

```bash
python generate_pipeline.py
```

This step:

- Generates all video clips using storyboard + camera scripts + keyframes
- Merges clips into a final MV
- Saves the result as result/{music_name}/mv_{music_name}.mp4

### 3. Output Directory Structure

After running the full pipeline, the output directory will contain:

```bash
result/{music_name}/
├── camera/                 # Camera directions for each MV segment
├── output/                  # Generated video clips for each segment
├── picture/                # First-frame images of each MV segment
├── piece/                   # Audio segments cut from the original song
├── {music_name}_vocals.wav  # Separated vocal audio (optional)
├── {music_name}.mp3         # The full original audio
├── label.json               # Character Bank
├── mv_{music_name}.mp4      # The final generated music video
├── name.txt                 # Full name of the song
└── story.json               # Complete MV storyboard
```

---

## 📊 Benchmark & Evaluation

We evaluate AutoMV with:

### **Objective Metric**

- **ImageBind Score (IB)** — cross-modal similarity
between audio and visual content
The relevant code is in  evaluate/IB.

### **LLM-Based Evaluation (12 Criteria)**

Using multimodal LLMs (Gemini-2.5-Pro/Flash) to score:

- Technical quality
- Post-production
- Music content alignment
- Artistic quality

The relevant code is in evaluate/LLM.

### **Human Expert Evaluation**

Music producers, MV directors, and industry practitioners scored each sub-criterion (1–5).

---

## 🧪 Experimental Results

On a benchmark of **30 professionally released songs**, AutoMV outperforms existing commercial systems:

| Method            | Cost   | Time     | IB ↑     | Human Score ↑ |
| ----------------- | ------ | -------- | -------- | ------------- |
| Revid.ai-base     | ~$10   | 5–10min  | 19.9     | 1.06          |
| OpenArt-story     | $20–40 | 10–20min | 18.5     | 1.45          |
| **AutoMV (ours)** | $10–20 | ~30min   | **24.4** | **2.42**      |
| Human (experts)   | ≥$10k  | Weeks    | 24.1     | 2.90          |

AutoMV greatly improves:

- Character consistency
- Shot continuity
- Audio–visual correlation
- Storytelling & theme relevance
- Overall coherence of long-form MVs

---

## 📚 Citation

If you use AutoMV in your research, please cite:

```bibtex
@article{AutoMV2026,
  title   = {AutoMV: An Automatic Multi-Agent System for Music Video Generation},
  author  = {Anonymous},
  journal = {arxiv},
  year    = {2025}
}
```

---

## 📝 License

This project is released under the MIT/BSD/Apache 2.0 License.
(Choose your license accordingly.)

---

## 🤝 Acknowledgements

AutoMV builds on:

- [Qwen-Wan 2.2 (lip-sync)](https://github.com/Wan-Video/Wan2.2)
- [Whisper](https://github.com/openai/whisper)
- [htdemucs](https://github.com/facebookresearch/demucs)
- [SongFormer](https://github.com/ASLP-lab/SongFormer)
