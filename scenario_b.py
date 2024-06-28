import os
import shutil
import boto3
import cv2

from utils import create_model, download_file_from_s3, process_image, untar_file


def scenario_B(
    s3: boto3.session.Session.resource,
    bucket_name: str,
    prefix: str,
    tar_name: str,
):
    """
    1. downloading the tar from s3 using boto3
    2. performing object detection
    """
    model = create_model()
    # Define local paths for download and extraction
    local_download_path = os.path.join("b", tar_name)
    local_extract_path = os.path.join("b", "out")

    os.makedirs(local_extract_path, exist_ok=True)

    # Download the tar file from S3
    download_file_from_s3(
        s3_client=s3,
        bucket_name=bucket_name,
        remote_filename=f"{prefix}/{tar_name}",
        local_path=local_download_path,
    )

    untar_file(local_download_path, local_extract_path)
    image_files = [f for f in os.listdir(local_extract_path) if f.endswith(".jp2")]
    for file in image_files:
        image_path = os.path.join(local_extract_path, file)
        image = cv2.imread(image_path)
        process_image(model=model, image=image)

    # clean up
    os.remove(local_download_path)
    shutil.rmtree(local_extract_path)

    return None
