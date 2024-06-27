import tarfile
import botocore
import cv2
from ultralytics import YOLO


def download_file_from_s3(s3_client, bucket_name, remote_filename, local_path):
    try:
        s3_client.download_file(bucket_name, remote_filename, local_path)
        print(f"Downloaded {remote_filename} from S3 to {local_path}")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(
                f"The object {remote_filename} does not exist in bucket {bucket_name}."
            )
        else:
            raise


def untar_file(tar_file: str, output_dir, mode="r"):
    with tarfile.open(tar_file, mode) as tar:
        tar.extractall(output_dir)
        print(f"Extracted contents of {tar_file} to {output_dir}")


def process_image(model: YOLO, image: cv2.typing.MatLike):
    results = model.predict(source=image, save=True)
    return results
