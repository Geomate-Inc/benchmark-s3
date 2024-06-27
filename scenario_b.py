import os
import boto3
import cv2

from utils import download_file_from_s3, process_image, untar_file


def scenario_B(
    s3: boto3.session.Session.client, bucket_name: str, prefix: str, tar_name: str
):
    """
    1. downloading the tar from s3 using boto3
    2. performing object detection
    """
    # Define local paths for download and extraction
    local_download_path = os.path.join("b", tar_name)
    local_extract_path = os.path.join("b", prefix)

    os.makedirs(local_download_path, exist_ok=True)
    os.makedirs(local_extract_path, exist_ok=True)

    # Download the tar file from S3
    download_file_from_s3(s3, bucket_name, f"{prefix}/{tar_name}", local_download_path)

    untar_file(local_download_path, local_extract_path)
    image_files = [f for f in os.listdir(local_extract_path) if f.endswith(".jp2")]
    for file in image_files:
        image_path = os.path.join(local_extract_path, file)
        image = cv2.imread(image_path)
        for result in range(process_image(image)):
            print(result)

    # clean up
    os.remove(local_download_path)
    os.rmdir(local_extract_path)
