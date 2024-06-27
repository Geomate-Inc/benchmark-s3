import boto3
import os
import cv2
import numpy as np
from ultralytics import YOLO

from utils import process_image


def scenario_A(
    s3: boto3.session.Session.client, bucket_name: str, prefix: str, model: YOLO
) -> None:
    """
    1. get images straight from s3 using boto3 and load one by one in memory and infer
    2. performing object detection
    """
    # Initialize the S3 client

    # List objects within the specified prefix
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Download the object to memory
            response = s3.get_object(Bucket=bucket_name, Key=key)
            file_content = response["Body"].read()
            image_array = np.frombuffer(file_content, np.uint8)

            # Decode NumPy array to OpenCV image format
            image_cv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            process_image(model=model, image=image_cv)

    return
