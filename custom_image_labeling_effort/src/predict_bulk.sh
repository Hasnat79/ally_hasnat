#!/usr/bin/env bash
set -e
MODEL="/data/hma18/ally_hasnat/custom_image_labeling_effort/src/runs_obb/cattle_y8n_obb10/weights/best.pt"
IN="/data/hma18/ally_hasnat/custom_image_labeling_effort/data/unlabeled_color_mapped_images"         # put new, unlabeled cattle images here
OUT="/data/hma18/ally_hasnat/custom_image_labeling_effort/data/autolabels"              # YOLO-OBB txts will be created here

yolo task=obb mode=predict \
  model="$MODEL" \
  source="$IN" \
  save_txt=True save_conf=True \
  project=runs_obb name=pred_cattle_y8n_obb

# Move generated labels to a clean folder
mkdir -p "$OUT"
find runs_obb/pred_cattle_y8n_obb/labels -name '*.txt' -exec cp {} "$OUT" \;

echo "Auto-labels written to $OUT/"