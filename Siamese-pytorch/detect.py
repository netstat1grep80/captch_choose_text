import yolov5
import torch
from siamese import Siamese

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image, ImageDraw
from siamese import Siamese
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_image_from_coordinates(image, coordinates):
    x1, y1, x2, y2 = coordinates
    return image.crop((x1, y1, x2, y2))

def load_image_from_url(image_path, cut_h_offset, clear_area):
    image = Image.open(image_path)
    width, height = image.size
    box = (0, 0, width, height - cut_h_offset)
    image = image.crop(box)
    draw = ImageDraw.Draw(image)
    draw.rectangle([tuple([0,0]), tuple(clear_area)], fill='white')
    #image.show()
    #image.save('url_image.png')
    return image


def plot_images_with_similarity(image, similarities):
    
    num_pairs = len(similarities)

    # 设置子图的行数和列数
    rows = 2
    cols = num_pairs

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))


    for i, (coord_small, coord_large, similarity) in enumerate(similarities):
        image_small = get_image_from_coordinates(image, coord_small)
        image_large = get_image_from_coordinates(image, coord_large)
        #image_small.show()
        #image_large.show()
        # 显示小图像
        axes[0, i].imshow(image_small)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"{i+1}")

        # 显示大图像
        axes[1, i].imshow(image_large)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"{i+1}\nSimilarity: {similarity.item()}")

    fig.tight_layout()
    #print(axes)
    plt.show()

SiameseModel = Siamese()
similarities = []

model = yolov5.load('./weights/captch.pt')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

image_path = './bilibili.jpg'
image = Image.open(image_path)
#image = load_image_from_url(image_path, 55, [135 ,50])
#crop_image_path = './corp.jpg'
#image.save(image_path)
# perform inference
results = model(image_path)
results = model(image_path, size=1280)
results = model(image_path, augment=True)

predictions = results.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2

#scores = predictions[:, 4]
#categories = predictions[:, 5]
print(boxes)
#results.show()

chars = []
targets = []

# 遍历坐标数据，根据条件将数据放入对应的数组中
for coord in boxes:
    if coord[1].item() > 330 and coord[3].item() > 330: 
        chars.append(coord.tolist())
    else:
        targets.append(coord.tolist())

chars.sort(key=lambda x: x[0])

boxes = boxes.to(device)

for coord_small in chars:
    image_small = get_image_from_coordinates(image, coord_small)
    max_similarity = 0
    best_coord_large = None
    for coord_large in targets:
        image_large = get_image_from_coordinates(image, coord_large)
        similarity =  SiameseModel.detect_image(image_small, image_large)
        if similarity > max_similarity:
            max_similarity = similarity
            best_coord_large = coord_large
    similarities.append((coord_small, best_coord_large, max_similarity))





plot_images_with_similarity(image ,similarities)
