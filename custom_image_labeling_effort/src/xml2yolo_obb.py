import os, glob, math, argparse
import xml.etree.ElementTree as ET
from pathlib import Path

def rotate_corners(cx, cy, bw, bh, theta):
    dx, dy = bw/2.0, bh/2.0
    c, s = math.cos(theta), math.sin(theta)
    local = [(-dx,-dy),(dx,-dy),(dx,dy),(-dx,dy)]
    return [(c*x - s*y + cx, s*x + c*y + cy) for (x,y) in local]

def order_clockwise(pts):
    # pts: list[(x,y)] in pixels; return CW order
    cx = sum(p[0] for p in pts)/4.0
    cy = sum(p[1] for p in pts)/4.0
    pts_with_ang = []
    for i,(x,y) in enumerate(pts):
        ang = math.atan2(y - cy, x - cx)
        pts_with_ang.append((ang, i, (x,y)))
    # sort by angle ascending then ensure clockwise
    pts_sorted = [p for _,_,p in sorted(pts_with_ang)]
    return pts_sorted

def inside_score(pts, W, H, margin=5.0):
    # how many corners are within the image (with small margin)
    ok = 0
    for x,y in pts:
        if -margin <= x <= W+margin and -margin <= y <= H+margin:
            ok += 1
    return ok

def polygon_area(pts):
    # Shoelace (pixels^2)
    area = 0.0
    for i in range(4):
        x1,y1 = pts[i]
        x2,y2 = pts[(i+1)%4]
        area += x1*y2 - x2*y1
    return abs(area) * 0.5

def best_theta(cx,cy,bw,bh,theta_raw,W,H):
    # Try interpreting theta as radians OR degrees; pick the one with more in-bounds corners.
    cands = [theta_raw, math.radians(theta_raw)]
    best = None; best_sc = -1
    for t in cands:
        pts = rotate_corners(cx,cy,bw,bh,t)
        sc = inside_score(pts, W, H)
        if sc > best_sc:
            best = (t, pts); best_sc = sc
    return best  # (theta, pts)

def parse_one(xml_path, out_dir, class_map):
    root = ET.parse(xml_path).getroot()
    fname = root.findtext("filename")
    W = int(root.find("size/width").text)
    H = int(root.find("size/height").text)
    objects = root.findall("object")

    lines = []
    for obj in objects:
        name = obj.findtext("name")
        if name not in class_map: 
            continue
        cls = class_map[name]
        rb = obj.find("robndbox")
        if rb is None:
            continue
        cx = float(rb.findtext("cx")); cy = float(rb.findtext("cy"))
        bw = float(rb.findtext("w"));  bh = float(rb.findtext("h"))
        theta_raw = float(rb.findtext("angle"))

        theta, pts_px = best_theta(cx,cy,bw,bh,theta_raw,W,H)
        pts_px = order_clockwise(pts_px)

        # Validate area; if tiny, skip (likely degenerate)
        if polygon_area(pts_px) < 10.0:   # 10 px^2 threshold
            continue

        # Normalize AFTER computing corners; clip at the very end
        pts_norm = []
        for (x,y) in pts_px:
            xn = max(0.0, min(1.0, x / W))
            yn = max(0.0, min(1.0, y / H))
            pts_norm.extend([xn, yn])

        # Sanity: x’s and y’s must vary
        xs = pts_norm[0::2]; ys = pts_norm[1::2]
        if max(xs)-min(xs) < 1e-6 or max(ys)-min(ys) < 1e-6:
            continue

        line = str(cls) + " " + " ".join(f"{v:.6f}" for v in pts_norm)
        lines.append(line)

    if not lines:
        return None

    outp = Path(out_dir)/ (Path(fname).stem + ".txt")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        f.write("\n".join(lines))
    return str(outp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--ann_xml", default="ann_xml")
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    class_map = {"cattle": 0}

    print(args.ann_xml)
    xmls = sorted(glob.glob(os.path.join(args.ann_xml, "*.xml")))
    print(f"Found {len(xmls)} XML files")
    kept = []
    for xp in xmls:
        out = parse_one(xp, args.labels, class_map)
        if out:
            kept.append(ET.parse(xp).getroot().findtext("filename"))

    # Split
    import random
    random.seed(args.seed)
    random.shuffle(kept)
    n = len(kept)
    n_val = max(1, int(n * args.val_ratio))
    val = set(kept[:n_val])

    Path("splits").mkdir(exist_ok=True)
    with open("splits/train.txt","w") as ftr, open("splits/val.txt","w") as fva:
        for im in kept:
            (fva if im in val else ftr).write(os.path.join(args.images, im) + "\n")

    with open("data.yaml","w") as f:
        f.write("""# auto-generated
path: .
train: splits/train.txt
val: splits/val.txt
names:
  0: cattle
""")
    print(f"Done. Train: {n-n_val}, Val: {n_val}")

if __name__ == "__main__":
    main()
