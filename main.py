import os
import boto3
from ultralytics import YOLO
from scenario_a import scenario_A
from scenario_b import scenario_B
from utils import create_s3_session


def main():
    session = create_s3_session()
    s3 = session.client("s3")
    a_prefix = "datasets_vertical/detroit-michigan-2022/city-images-SR/mar/tiles/10_percent/1250_150/"
    bench_prefix = "bench"
    bucket_name = "geomate-data-repo-dev"

    scenario_A(s3=s3, bucket_name=bucket_name, prefix=a_prefix)

    scenario_B(
        s3=s3,
        bucket_name=bucket_name,
        prefix=bench_prefix,
        tar_name="detroit.tar",
    )


if __name__ == "__main__":
    main()
