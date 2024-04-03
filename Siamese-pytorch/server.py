import yolov5
import torch
from siamese import Siamese
from flask import Flask, request, jsonify
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from siamese import Siamese
import random
import string
import io
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

def get_image_from_coordinates(image, coordinates):
    x1, y1, x2, y2 = coordinates
    return image.crop((x1, y1, x2, y2))

def load_image_from_bytes(image_bytes, cut_h_offset, clear_area):
    image = Image.open(io.BytesIO(image_bytes))
    if cut_h_offset > 0 :
        width, height = image.size
        box = (0, 0, width, height - cut_h_offset)
        image = image.crop(box)
        draw = ImageDraw.Draw(image)
        draw.rectangle([tuple([0,0]), tuple(clear_area)], fill='white')
    #image.show()
    return image


def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

def plot_images_with_similarity(image, similarities,resutl_image):
    
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
    plt.savefig(resutl_image)
    plt.close()

SiameseModel = Siamese()


model = yolov5.load('./weights/captch.pt')
  
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image



@app.route('/detect_image', methods=['POST'])
def detect_image():
    similarities = []
    filename = generate_random_string(10)
    post_image = request.files['image'].read()
    result_image_path = './tmp/result_' + filename + '.png'
    image = load_image_from_bytes(post_image, 50, [135 ,50])
    crop_image_path = './tmp/cropt_'+ filename + '.png'
    image.save(crop_image_path)
    # perform inference
    results = model(crop_image_path)
    width, height = image.size
    results = model(crop_image_path, size=width)
    results = model(crop_image_path, augment=True)

    #if os.path.exists(crop_image_path):
     #   os.remove(crop_image_path)

    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    errcode = 0
    msg = ''
    print(boxes)
    
    if len(boxes) % 2 != 0:
        msg = 'to many objects'
        errcode = 10001
        return jsonify({'code':errcode, 'msg': msg, 'result_image_path': result_image_path})
    chars = []
    targets = []
    result_coord = []

    # 遍历坐标数据，根据条件将数据放入对应的数组中
    for coord in boxes:
        if coord[1].item() < 50 and coord[3].item() < 50:
            chars.append(coord.tolist())
        else:
            targets.append(coord.tolist())

    chars.sort(key=lambda x: x[0])

    boxes = boxes.to(device)
    

    if len(chars) == 0:
        errcode = 10002
        msg = 'not detect any object'
        return jsonify({'code':errcode,'result_image_path': result_image_path})
    #print(chars)
    for coord_small in chars:
        image_small = get_image_from_coordinates(image, coord_small)
        max_similarity = 0
        max_target = []

        best_coord_large = None
        for coord_large in targets:
            image_large = get_image_from_coordinates(image, coord_large)
            similarity =  SiameseModel.detect_image(image_small, image_large)
            if similarity > max_similarity:
                max_similarity = similarity
                best_coord_large = coord_large
                max_target = coord_large
        print(max_target)
        result_coord.append(max_target)
        similarities.append((coord_small, best_coord_large, max_similarity))

    #print(similarities)
    plot_images_with_similarity(image ,similarities,result_image_path)
    return jsonify({'code':errcode,'result_image_path': result_image_path,'coord':result_coord})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)