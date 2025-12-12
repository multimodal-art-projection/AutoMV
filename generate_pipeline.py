from video_generate.video_generate_pipeline import full_video_gen
from generate_lip_video.gen_lip_sycn_video import gen_lip_sync_video
from generate_lip_video.gen_lip_sycn_video_jimeng import gen_lip_sync_video_jimeng
from config import Config

if __name__ == "__main__":
    music_video_name = "让浪漫做主"
    # Use Wan2.2 model. Slow but much more cheaper! A music video takes about 3-4 hours to generate in an A800 GPU.
    # gen_lip_sync_video(music_video_name)
    # Use Jimeng model. Fast but much more expensive! A music video takes about 15-20 minutes to generate
    gen_lip_sync_video_jimeng(music_video_name, config = Config)
    full_video_gen(music_video_name, config = Config)
    # https://github.com/bupterlxp/AutoMV.git