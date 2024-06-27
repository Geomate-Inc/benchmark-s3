import os
import boto3
from ultralytics import YOLO
from scenario_a import scenario_A
from scenario_b import scenario_B
from scenario_c import scenario_C


def main():
    access_key_id = os.environ.get("ACCESS_KEY_ID")
    access_key_secret = os.environ.get("ACCESS_KEY_SECRET")
    s3 = boto3.client(
        "s3", aws_access_key_id=access_key_id, aws_secret_access_key=access_key_secret
    )
    a_prefix = "/datasets_vertical/"
    bench_prefix = "/bench/"
    bucket_name = ""
    model = YOLO("yolov8n.pt")

    exit()

    scenario_A(s3=s3, bucket_name=bucket_name, model=model, prefix=a_prefix)

    scenario_B(
        s3=s3, bucket_name=bucket_name, prefix=bench_prefix, tar_name="detroit.tar"
    )
    scenario_C(
        s3=s3, bucket_name=bucket_name, prefix=bench_prefix, tar_gz_name="detroit.tar"
    )


if __name__ == "__main__":
    main()
