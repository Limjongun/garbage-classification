Garbage Classification using YOLO

A simple AI project for classifying garbage using YOLO and Python.
Overview

This project uses a YOLO-based model to classify different types of garbage from images.
It can be used for basic waste management systems and recycling automation.

Project Structure
garbage-classification/
│── datasets/              # Dataset for training (not fully included)
│── runs/                  # Training results (ignored)
│── imageee.py             # Image classification script
│── train.py               # Training script
Requirements
Python 3.x
OpenCV
PyTorch
Ultralytics (YOLO)

Install dependencies:

pip install -r requirements.txt
Usage

Run classification:

python imageee.py

Train model:

python train.py
Notes
Dataset folder may not include all images due to size limitations
Training results (runs/) are not required for running the project
Model files (.pt) are not included Future Improvements
Improve dataset quality and size
Add real-time detection (webcam)
Deploy as web or mobile application
