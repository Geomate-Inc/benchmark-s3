import os
import sys
import tarfile
import threading
import boto3
import botocore
import cv2
from ultralytics import YOLO


def download_file_from_s3(
    s3_client: boto3.session.Session.client,
    bucket_name: str,
    remote_filename: str,
    local_path: str,
):
    s3_client.download_file(
        bucket_name,
        remote_filename,
        local_path,
        Callback=ProgressPercentage(remote_filename),
    )

    print(f"Downloaded {remote_filename} from S3 to {local_path}")


def untar_file(tar_file: str, output_dir, mode="r"):
    with tarfile.open(tar_file, mode) as tar:
        tar.extractall(output_dir)
        print(f"Extracted contents of {tar_file} to {output_dir}")


def process_image(model: YOLO, image: cv2.typing.MatLike):
    results = model.predict(source=image, save=True)
    return results


class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            sys.stdout.write(
                "\r%s --> %s bytes transferred" % (self._filename, self._seen_so_far)
            )
            sys.stdout.flush()


def create_s3_session():
    access_key_id = os.environ.get("ACCESS_KEY_ID")
    access_key_secret = os.environ.get("ACCESS_KEY_SECRET")
    session = boto3.session.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=access_key_secret,
        region_name="us-east-1",
    )
    return session


def create_model():
    model = YOLO("yolov8n.pt")
    return model
