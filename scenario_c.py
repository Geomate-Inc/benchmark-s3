import os

import boto3
import cv2

from utils import download_file_from_s3, process_image, untar_file


def scenario_C(
    s3: boto3.session.Session.client, bucket_name: str, prefix: str, tar_gz_name: str
):
    """
    1. downloading the tar.gz from s3 using boto3
    2. performing object detection
    """

    # Define local paths for download and extraction
    local_download_path = os.path.join("c", tar_gz_name)
    local_extract_path = os.path.join("c", prefix)

    os.makedirs(local_download_path, exist_ok=True)
    os.makedirs(local_extract_path, exist_ok=True)

    # Download the tar file from S3
    download_file_from_s3(
        s3, bucket_name, f"{prefix}/{tar_gz_name}", local_download_path
    )

    untar_file(local_download_path, local_extract_path, mode="r:gz")
    image_files = [f for f in os.listdir(local_extract_path) if f.endswith(".jp2")]
    for file in image_files:
        image_path = os.path.join(local_extract_path, file)
        image = cv2.imread(image_path)
        for result in range(process_image(image)):
            print(result)

    # clean up
    os.remove(local_download_path)
    os.rmdir(local_extract_path)
