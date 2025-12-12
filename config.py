from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    DOUBAO_API_KEY = os.getenv('DOUBAO_API_KEY')
    ALIYUN_ID = os.getenv('ALIYUN_OSS_ACCESS_KEY_ID')
    ALIYUN_SECRET = os.getenv('ALIYUN_OSS_ACCESS_KEY_SECRET')
    ALIYUN_OSS_BUCKET_NAME = os.getenv('ALIYUN_OSS_BUCKET_NAME')
    HUOSHAN_ACCESS_KEY = os.getenv('HUOSHAN_ACCESS_KEY')
    HUOSHAN_SECRET_KEY = os.getenv('HUOSHAN_SECRET_KEY')
    GPU_ID = os.getenv('GPU_ID', 0)
    whipser_model = os.getenv('WHISPER_MODEL', 'openai/whisper-large-v2')
    qwen_model = os.getenv('QWEN_OMNI_MODEL', 'Qwen/Qwen2.5-Omni-7B')
    music_name = "1"
    @classmethod
    def validate(cls):
        """验证必需的配置是否存在"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env")
        if not cls.DOUBAO_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in .env")

if __name__ == "__main__":
    Config.validate()
    print("GEMINI_API_KEY:", Config.GEMINI_API_KEY)