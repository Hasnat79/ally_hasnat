#!/usr/bin/env bash
set -e
# Ultralytics CLI expects: labels in ./labels, images referenced by splits/train.txt & val.txt
# Start with a nano model and ~50-100 epochs; bump if underfitting.

yolo task=obb mode=train \
  model=yolov8n-obb.pt \
  data=/data/hma18/ally_hasnat/custom_image_labeling_effort/src/data.yaml\
  epochs=80 \
  imgsz=640 \
  batch=16 \
  lr0=0.002 \
  patience=20 \
  cos_lr=True \
  project=runs_obb name=cattle_y8n_obb

# Best weights will be at runs_obb/cattle_y8n_obb/weights/best.pt
