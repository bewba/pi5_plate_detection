from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("licence-plate-3jqkb").project("license-plate-az47f")
version = project.version(1)
dataset = version.download("yolov11")
                