#Import libraries
import xml.etree.ElementTree as ET
import os
import random
import shutil

#show path
img_dir = "../database/images"
xml_dir = "../database/annotations"

base = "database"

# correct YOLO structure
train_img = f"{base}/images/train"
val_img = f"{base}/images/val"

train_lbl = f"{base}/labels/train"
val_lbl = f"{base}/labels/val"

for p in [train_img, val_img, train_lbl, val_lbl]:
    os.makedirs(p, exist_ok=True)


# XML → YOLO conversion
def convert(xml_file, txt_file):
#to load xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()
#to find size,width,height
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
#open file
    with open(txt_file, 'w') as f:
#find all objects
        for obj in root.findall('object'):
            cls = 0
            box = obj.find('bndbox')
            
            xmin = int(box.find('xmin').text)
            xmax = int(box.find('xmax').text)
            ymin = int(box.find('ymin').text)
            ymax = int(box.find('ymax').text)

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h

            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            f.write(f"{cls} {x_center} {y_center} {bw} {bh}\n")


images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]

random.shuffle(images)

split = int(len(images) * 0.8)

train_files = images[:split]
val_files = images[split:]


# process function (IMPORTANT)
def process(files, img_out, lbl_out):

    for img in files:

        name = os.path.splitext(img)[0]

        img_path = os.path.join(img_dir, img)
        xml_path = os.path.join(xml_dir, name + ".xml")
        txt_path = os.path.join(lbl_out, name + ".txt")

        # convert xml → txt
        convert(xml_path, txt_path)

        # copy image
        shutil.copy(img_path, os.path.join(img_out, img))


# run pipeline
process(train_files, train_img, train_lbl)
process(val_files, val_img, val_lbl)