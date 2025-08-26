# viz_check_safe.py
import cv2, os, glob
import numpy as np
from pathlib import Path
import argparse

IM_DIR = "../data/labeled_color_mapped_images"
LB_DIR = "/data/hma18/ally_hasnat/custom_image_labeling_effort/data/labeled_color_mapped_images/"
OUT_DIR = "viz"
Path(OUT_DIR).mkdir(exist_ok=True)
parser = argparse.ArgumentParser(description="Visualize labeled polygons on images.")
parser.add_argument("--im_dir", type=str, default="../data/labeled_color_mapped_images", help="Directory with input images")
parser.add_argument("--lb_dir", type=str, default="/data/hma18/ally_hasnat/custom_image_labeling_effort/data/labeled_color_mapped_images/", help="Directory with label txt files")
parser.add_argument("--out_dir", type=str, default="viz", help="Directory to save output visualizations")
args = parser.parse_args()

IM_DIR = args.im_dir
LB_DIR = args.lb_dir
OUT_DIR = args.out_dir
Path(OUT_DIR).mkdir(exist_ok=True)
def denorm(poly_norm, w, h):
    pts = np.array(poly_norm, dtype=float).reshape(-1,2)
    pts[:,0] *= w; pts[:,1] *= h
    return pts.astype(int)


for lb in glob.glob(os.path.join(LB_DIR, "*.txt")):
    stem = Path(lb).stem
    imgp = None
    print(f"Looking for image for {stem}")
    for ext in (".png",".jpg",".jpeg"):
        cand = os.path.join(IM_DIR, stem+ext)
        if os.path.exists(cand):
            imgp = cand; break
            print(imgp)
            exit()
    if imgp is None: continue
    img = cv2.imread(imgp)
    if img is None: continue
    h, w = img.shape[:2]
    for line in open(lb):
        parts = line.strip().split()
        if len(parts) < 9: continue
        # allow optional confidences at the end
        coords = parts[1:9]
        pts = denorm(coords, w, h)
        cv2.polylines(img, [pts], True, (0,255,0), 3)
        for (x,y) in pts:
            cv2.circle(img, (x,y), 3, (255,0,0), -1)
    cv2.imwrite(os.path.join(OUT_DIR, stem+".jpg"), img)
