# Object Detection Benchmark Project

This project performs benchmarking for object detection using YOLO (You Only Look Once) with images sourced from Amazon S3. It includes two scenarios:

## Scenarios

### Scenario A: Direct Image Inference from S3

- **Objective**: Perform object detection directly on images retrieved from an S3 bucket using boto3.
- **Steps**:
  1. Initialize an S3 session and client.
  2. Iterate through objects in a specified S3 prefix.
  3. Download each image in memory, decode it using OpenCV, and perform object detection using YOLO.
  4. Clean up temporary files.

### Scenario B: Tar File Extraction and Inference

- **Objective**: Download a tar file containing images from S3, extract them locally, and perform object detection.
- **Steps**:
  1. Download a tar file from S3.
  2. Extract the tar file locally.
  3. Iterate through JPEG 2000 images extracted from the tar file.
  4. Read each image using OpenCV, perform object detection using YOLO
  5. Clean up temporary files.

## Setup

To run this project, ensure you have the necessary dependencies installed using Poetry:

- Python 3.x
- Poetry (Install Poetry: https://python-poetry.org/docs/#installation)

## Configuration

Ensure you have AWS credentials set up with access to the specified S3 bucket. Set the following environment variables:

- `ACCESS_KEY_ID`: Your AWS access key ID.
- `ACCESS_KEY_SECRET`: Your AWS secret access key.

## Usage

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Run the benchmarking script:
   ```bash
   poetry run python -m cProfile -o test.prof main.py | tee output.log
   ```

2. Run the benchmarking script:
   ```bash
   poetry run snakeviz test.prof
   ```
