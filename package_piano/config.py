import os
from dotenv import load_dotenv

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_DATA_PREFIX = os.getenv("GCS_DATA_PREFIX")

LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "data")
LOCAL_RAW_DIR = os.getenv("LOCAL_RAW_DIR", "raw_data")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
