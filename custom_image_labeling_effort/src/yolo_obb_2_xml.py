import os, glob, cv2, math
import numpy as np
from pathlib import Path
from lxml import etree as ET
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Convert YOLO-OBB labels to XML format.")
parser.add_argument("--im_dir", type=str, required=True, help="Directory with images")
parser.add_argument("--lb_dir", type=str, required=True, help="Directory with YOLO-OBB .txt labels")
parser.add_argument("--out_xml", type=str, required=True, help="Output directory for XML files")
args = parser.parse_args()

IM_DIR = args.im_dir
LB_DIR = args.lb_dir
OUT_XML = args.out_xml

Path(OUT_XML).mkdir(exist_ok=True)

def make_xml(fname, w, h, objs):
    root = ET.Element("annotation", verified="yes")
    ET.SubElement(root, "folder").text = str(Path(fname).parent.name)
    ET.SubElement(root, "filename").text = Path(fname).name
    ET.SubElement(root, "path").text = str(Path(fname).resolve())

    source = ET.SubElement(root, "source"); ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    for (cls_name, cx, cy, bw, bh, angle_rad) in objs:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = cls_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        rb = ET.SubElement(obj, "robndbox")
        ET.SubElement(rb, "cx").text = f"{cx:.6f}"
        ET.SubElement(rb, "cy").text = f"{cy:.6f}"
        ET.SubElement(rb, "w").text = f"{bw:.6f}"
        ET.SubElement(rb, "h").text = f"{bh:.6f}"
        ET.SubElement(rb, "angle").text = f"{angle_rad:.6f}"
        ET.SubElement(obj, "extra")
    return root

def rect_from_4pts(pts):
    # pts: 4x2 in pixels
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (cx, cy), (w, h), angle_deg = rect
    # OpenCV angle is [-90, 0); convert to radians (counter-clockwise positive)
    angle_rad = math.radians(angle_deg)
    return cx, cy, w, h, angle_rad
lb_files = glob.glob(os.path.join(LB_DIR, "*.txt"))
for lb in tqdm(lb_files, desc="Converting YOLO-OBB to XML"):
    stem = Path(lb).stem
    # find image
    imgp = None
    for ext in (".png",".jpg",".jpeg"):
        cand = os.path.join(IM_DIR, stem+ext)
        if os.path.exists(cand):
            imgp = cand; break
    if imgp is None: continue
    img = cv2.imread(imgp); h, w = img.shape[:2]

    objs = []
    for line in open(lb):
        parts = line.strip().split()
        if len(parts) < 9: continue
        cls_id = int(parts[0])
        # single class: "cattle"
        cls_name = "cattle"
        coords = np.array(list(map(float, parts[1:9]))).reshape(4,2)
        # denormalize
        pts = coords.copy()
        pts[:,0] *= w; pts[:,1] *= h
        cx, cy, bw, bh, ang = rect_from_4pts(pts)
        objs.append((cls_name, cx, cy, bw, bh, ang))

    xml_root = make_xml(imgp, w, h, objs)
    xml_str = ET.tostring(xml_root, pretty_print=True, encoding="UTF-8", xml_declaration=True)
    with open(os.path.join(OUT_XML, stem+".xml"), "wb") as f:
        f.write(xml_str)

print(f"XMLs written to {OUT_XML}")
