import os
import subprocess
from picture_generate.picture import generate_one_mv
from config import Config

if __name__ == "__main__":
    music_name = Config.music_name
    base_dir = os.path.abspath(os.path.dirname(__file__)) 
    result_dir = os.path.abspath(os.path.join(base_dir, "..", "result", music_name))
    os.makedirs(result_dir, exist_ok=True)

    audio_path = os.path.join(result_dir, f"{music_name}.mp3")
    scp_path = os.path.join(result_dir, f"{music_name}.scp")
    output_dir = os.path.join(result_dir)
    scp_line = f"{audio_path}\n"

    with open(scp_path, "w", encoding="utf-8") as f:
        f.write(scp_line)
    print(f"[INFO] Wrote SCP file: {scp_path}")
    print(f"[INFO] Content: {scp_line}")

    infer_cmd = [
        "python",
        "infer/infer.py",
        "-i", scp_path,
        "-o", output_dir,
        "--model", "SongFormer",
        "--checkpoint", "SongFormer.safetensors",
        "--config_path", "SongFormer.yaml",
        "-gn", "1",
        "-tn", "1"
    ]

    result = subprocess.run(
        infer_cmd,
        cwd="picture_generate/SongFormer/src/SongFormer",
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.path.abspath(os.path.join(os.path.dirname(__file__), "SongFormer", "src", "SongFormer")) + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")}
    )

    print("STDOUT:\n", result.stdout)
    print("stderr:\n", result.stderr)
    print("returncode:", result.returncode)


    
    success = False
    while not success:
        try:
            # Generate a single music video. Parameters are (in order): music video name, starting index of the segment to be regenerated, ending index of the segment to be regenerated
            generate_one_mv(music_name)
            success = True 
        except Exception as e:
            print(f"generate_one_mv failed, retrying... Error message: {e}")
