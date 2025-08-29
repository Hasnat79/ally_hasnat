
import os
import argparse
import xml.etree.ElementTree as ET
import math
from PIL import Image

def parse_args():
	parser = argparse.ArgumentParser(description="Crop images using bounding boxes from XML labels.")
	parser.add_argument('--img_dir', required=True, help='Directory with PNG images')
	parser.add_argument('--label_dir', required=True, help='Directory with XML label files')
	parser.add_argument('--output_dir', required=True, help='Directory to save cropped images')
	return parser.parse_args()


def parse_robndboxes(xml_path):
	tree = ET.parse(xml_path)
	root = tree.getroot()
	robndboxes = []
	for obj in root.findall('object'):
		robndbox = obj.find('robndbox')
		if robndbox is not None:
			cx = float(robndbox.find('cx').text)
			cy = float(robndbox.find('cy').text)
			w = float(robndbox.find('w').text)
			h = float(robndbox.find('h').text)
			angle = float(robndbox.find('angle').text)
			robndboxes.append((cx, cy, w, h, angle))
	return robndboxes


def crop_rotated(img, cx, cy, w, h, angle):
	# Convert angle from radians to degrees, anticlockwise for PIL
	angle_deg = math.degrees(angle)
	# Rotate image around center
	img_rot = img.rotate(angle_deg, center=(cx, cy), expand=True)
	# Calculate new center after rotation
	# Find offset due to expand=True
	w_img, h_img = img.size
	w_rot, h_rot = img_rot.size
	dx = (w_rot - w_img) / 2
	dy = (h_rot - h_img) / 2
	# The new center in rotated image
	cx_new = cx + dx
	cy_new = cy + dy
	# Crop axis-aligned rectangle around the rotated box
	left = int(cx_new - w/2)
	upper = int(cy_new - h/2)
	right = int(cx_new + w/2)
	lower = int(cy_new + h/2)
	return img_rot.crop((left, upper, right, lower))

def main():
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)
	for img_name in os.listdir(args.img_dir):
		if not img_name.lower().endswith('.png'):
			continue
		base_name = os.path.splitext(img_name)[0]
		xml_path = os.path.join(args.label_dir, base_name + '.xml')
		img_path = os.path.join(args.img_dir, img_name)
		if not os.path.exists(xml_path):
			print(f"Warning: No XML for {img_name}")
			continue
		robndboxes = parse_robndboxes(xml_path)
		img = Image.open(img_path)
		for i, (cx, cy, w, h, angle) in enumerate(robndboxes):
			cropped = crop_rotated(img, cx, cy, w, h, angle)
			out_name = f"{base_name}_crop_{i+1}.png"
			cropped.save(os.path.join(args.output_dir, out_name))
		img.close()

if __name__ == "__main__":
	main()
