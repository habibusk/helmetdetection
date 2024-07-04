# Load model directly
import os
from dotenv import load_dotenv
load_dotenv()
model_name = os.getenv('MODEL_NAME') # model name for huggingface.co in .env file
from transformers import AutoImageProcessor, AutoModelForObjectDetection
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name)