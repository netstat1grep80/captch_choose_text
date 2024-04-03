
## 一、yolov5训练
然后安装labelImg,开始打标（300张左右，很累）


提示的字定义成char，背景中的大字定义成target，实际上是两个标签，叫什么都行
完成打标工作后，打开trains.py找到这段参数设置

parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")

可以看到，默认使用的是data/coco128.yaml的配置，那么把这个文件复制到 ./dataset/traindata.yaml


编辑这个配置文件
```
path: d:/ai/captcha/yolov5/dataset
train: images
val: images
# Classes
names:
	 0: target
	 1: char

因为接下来要使用yolov5s.pt这个基础模型，所以要修改相应的models/yolov5s.yaml中的配置

# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# Parameters
nc: 2 # number of classes ,这个数字一定要和之前的traindata.yaml中的分类数一样
depth_multiple: 0.33 # model depth multiple
width_multiple: 0.50 # layer channel multiple
anchors:
	 - [10, 13, 16, 30, 33, 23] # P3/8
	 - [30, 61, 62, 45, 59, 119] # P4/16
	 - [116, 90, 156, 198, 373, 326] # P5/32
# YOLOv5 v6.0 backbone
backbone:
	 # [from, number, module, args]
	 [
	 	 [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
	 	 [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
	 	 [-1, 3, C3, [128]],
	 	 [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
	 	 [-1, 6, C3, [256]],
	 	 [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
	 	 [-1, 9, C3, [512]],
	 	 [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
	 	 [-1, 3, C3, [1024]],
	 	 [-1, 1, SPPF, [1024, 5]], # 9
	 ]
# YOLOv5 v6.0 head
head: [
	 	 [-1, 1, Conv, [512, 1, 1]],
	 	 [-1, 1, nn.Upsample, [None, 2, "nearest"]],
	 	 [[-1, 6], 1, Concat, [1]], # cat backbone P4
	 	 [-1, 3, C3, [512, False]], # 13
	 	 [-1, 1, Conv, [256, 1, 1]],
	 	 [-1, 1, nn.Upsample, [None, 2, "nearest"]],
	 	 [[-1, 4], 1, Concat, [1]], # cat backbone P3
	 	 [-1, 3, C3, [256, False]], # 17 (P3/8-small)
	 	 [-1, 1, Conv, [256, 3, 2]],
	 	 [[-1, 14], 1, Concat, [1]], # cat head P4
	 	 [-1, 3, C3, [512, False]], # 20 (P4/16-medium)
	 	 [-1, 1, Conv, [512, 3, 2]],
	 	 [[-1, 10], 1, Concat, [1]], # cat head P5
	 	 [-1, 3, C3, [1024, False]], # 23 (P5/32-large)
	 	 [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
	 ]
```
最后一步，开始执行训练脚本 ,workers如果太大内存会不够用，32g的内存workers也就能设置成2
```
python .\train.py --img 640 --batch 64 --epochs 600 --data	 .\dataset\traindata.yaml --weights yolov5s.pt --nosave --cache --device 0 --workers 2
```

执行后效果如下：
```
(captcha) PS D:\ai\captcha\yolov5> python .\train.py --img 640 --batch 64 --epochs 600 --data	 .\dataset\traindata.yaml --weights yolov5s.pt --nosave --cache --device 0 --workers 2
train: weights=yolov5s.pt, cfg=, data=.\dataset\traindata.yaml, hyp=data\hyps\hyp.scratch-low.yaml, epochs=600, batch_size=64, imgsz=640, rect=False, resume=False, nosave=True, noval=False, noautoanchor=False, noplots=False, evolve=None, evolve_population=data\hyps, resume_evolve=None, bucket=, cache=ram, image_weights=False, device=0, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=2, project=runs\train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest, ndjson_console=False, ndjson_file=False
github: skipping check (offline), for updates see https://github.com/ultralytics/yolov5
YOLOv5	 v7.0-294-gdb125a20 Python-3.10.14 torch-2.2.0+cu121 CUDA:0 (Quadro M5000, 8192MiB)
hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5	 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs\train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=2
	 	 	 	 	 	 	 		 from	 n	 	 params	 module	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 arguments
	 0	 	 	 	 	 	 	 	 -1	 1	 	 	 3520	 models.common.Conv	 	 	 	 	 	 	 	 	 	 	 [3, 32, 6, 2, 2]
	 1	 	 	 	 	 	 	 	 -1	 1	 		 18560	 models.common.Conv	 	 	 	 	 	 	 	 	 	 	 [32, 64, 3, 2]
	 2	 	 	 	 	 	 	 	 -1	 1	 		 18816	 models.common.C3	 	 	 	 	 	 	 	 	 	 	 	 [64, 64, 1]
	 3	 	 	 	 	 	 	 	 -1	 1	 		 73984	 models.common.Conv	 	 	 	 	 	 	 	 	 	 	 [64, 128, 3, 2]
	 4	 	 	 	 	 	 	 	 -1	 2	 	 115712	 models.common.C3	 	 	 	 	 	 	 	 	 	 	 	 [128, 128, 2]
	 5	 	 	 	 	 	 	 	 -1	 1	 	 295424	 models.common.Conv	 	 	 	 	 	 	 	 	 	 	 [128, 256, 3, 2]
	 6	 	 	 	 	 	 	 	 -1	 3	 	 625152	 models.common.C3	 	 	 	 	 	 	 	 	 	 	 	 [256, 256, 3]
	 7	 	 	 	 	 	 	 	 -1	 1		 1180672	 models.common.Conv	 	 	 	 	 	 	 	 	 	 	 [256, 512, 3, 2]
	 8	 	 	 	 	 	 	 	 -1	 1		 1182720	 models.common.C3	 	 	 	 	 	 	 	 	 	 	 	 [512, 512, 1]
	 9	 	 	 	 	 	 	 	 -1	 1	 	 656896	 models.common.SPPF	 	 	 	 	 	 	 	 	 	 	 [512, 512, 5]
 10	 	 	 	 	 	 	 	 -1	 1	 	 131584	 models.common.Conv	 	 	 	 	 	 	 	 	 	 	 [512, 256, 1, 1]
 11	 	 	 	 	 	 	 	 -1	 1	 	 	 		 0	 torch.nn.modules.upsampling.Upsample	 	 [None, 2, 'nearest']
 12	 	 	 	 		 [-1, 6]	 1	 	 	 		 0	 models.common.Concat	 	 	 	 	 	 	 	 	 	 [1]
 13	 	 	 	 	 	 	 	 -1	 1	 	 361984	 models.common.C3	 	 	 	 	 	 	 	 	 	 	 	 [512, 256, 1, False]
 14	 	 	 	 	 	 	 	 -1	 1	 		 33024	 models.common.Conv	 	 	 	 	 	 	 	 	 	 	 [256, 128, 1, 1]
 15	 	 	 	 	 	 	 	 -1	 1	 	 	 		 0	 torch.nn.modules.upsampling.Upsample	 	 [None, 2, 'nearest']
 16	 	 	 	 		 [-1, 4]	 1	 	 	 		 0	 models.common.Concat	 	 	 	 	 	 	 	 	 	 [1]
 17	 	 	 	 	 	 	 	 -1	 1	 		 90880	 models.common.C3	 	 	 	 	 	 	 	 	 	 	 	 [256, 128, 1, False]
 18	 	 	 	 	 	 	 	 -1	 1	 	 147712	 models.common.Conv	 	 	 	 	 	 	 	 	 	 	 [128, 128, 3, 2]
 19	 	 	 	 	 [-1, 14]	 1	 	 	 		 0	 models.common.Concat	 	 	 	 	 	 	 	 	 	 [1]
 20	 	 	 	 	 	 	 	 -1	 1	 	 296448	 models.common.C3	 	 	 	 	 	 	 	 	 	 	 	 [256, 256, 1, False]
 21	 	 	 	 	 	 	 	 -1	 1	 	 590336	 models.common.Conv	 	 	 	 	 	 	 	 	 	 	 [256, 256, 3, 2]
 22	 	 	 	 	 [-1, 10]	 1	 	 	 		 0	 models.common.Concat	 	 	 	 	 	 	 	 	 	 [1]
 23	 	 	 	 	 	 	 	 -1	 1		 1182720	 models.common.C3	 	 	 	 	 	 	 	 	 	 	 	 [512, 512, 1, False]
 24	 	 	 [17, 20, 23]	 1	 		 18879	 models.yolo.Detect	 	 	 	 	 	 	 	 	 	 	 [2, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 214 layers, 7025023 parameters, 7025023 gradients, 16.0 GFLOPs
Transferred 343/349 items from yolov5s.pt
AMP: checks passed
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
train: Scanning D:\ai\captcha\yolov5\dataset\labels.cache... 299 images, 0 backgrounds, 0 corrupt: 100%|
显示这一行最起码说你路径设置正确了
██████████| 299
train: Caching images (0.3GB ram): 100%|██████████| 299/299 [00:00<00:00, 511.06it/s]
val: Scanning D:\ai\captcha\yolov5\dataset\labels.cache... 299 images, 0 backgrounds, 0 corrupt: 100%|██████████| 299/2
val: Caching images (0.3GB ram): 100%|██████████| 299/299 [00:00<00:00, 1340.81it/s]
AutoAnchor: 6.29 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset
Plotting labels to runs\train\exp6\labels.jpg...
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to runs\train\exp6
Starting training for 600 epochs...

训练完成后的输出：

Epoch	 	 GPU_mem		 box_loss		 obj_loss		 cls_loss	 Instances	 	 		 Size
	 	 599/599	 	 	 14.2G	 	 0.01101	 	 0.01625	 5.667e-05	 	 	 	 332	 	 	 	 640: 100%|██████████| 5/5 [01:02<00:00, 12.47s/it]
	 	 	 	 	 	 	 		 Class	 		 Images	 Instances	 	 	 	 	 P	 	 	 	 	 R	 	 	 mAP50		 mAP50-95: 100%|██████████| 3/3 [00:10<00:00,	 3.56s/it]
	 	 	 	 	 	 	 	 		 all	 	 	 	 299	 	 		 1715	 	 	 	 	 1	 	 	 	 	 1	 	 	 0.995	 	 		 0.95
600 epochs completed in 12.046 hours.
Optimizer stripped from runs\train\exp6\weights\last.pt, 14.4MB
Results saved to runs\train\exp6
```

从bilibili找到一张登录用的图片进行测试.拷贝到data\images\下
```
python .\detect.py --weights .\runs\train\exp6\weights\last.pt --img 640 --conf 0.25 --source .\data\images\

detect: weights=['.\\runs\\train\\exp6\\weights\\last.pt'], source=.\data\images\, data=data\coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_csv=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs\detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
YOLOv5	 v7.0-294-gdb125a20 Python-3.10.14 torch-2.2.0+cu121 CUDA:0 (Quadro M5000, 8192MiB)
Fusing layers...
Model summary: 157 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
image 1/1 D:\ai\captcha\yolov5\data\images\bilibili.png: 640x512 4 targets, 14 chars, 67.0ms
Speed: 1.0ms pre-process, 67.0ms inference, 494.3ms NMS per image at shape (1, 3, 640, 640)
Results saved to runs\detect\exp
```

## 二、孪生神经网络（Siamese network）比较图片相似性

对之前打过点的图片进行切割，代码如下
```
import os
import cv2
 
def txt_cut_image(image_path,txt_path, new_image_path):
	 	 # 遍历所有图片
	 	 for file in os.listdir(image_path):
	 	 	 	 	 	 image_file_path = os.path.join(image_path, file)
 
	 	 	 	 	 	 # 构造对应的txt文件路径
	 	 	 	 	 	 txt_file = os.path.splitext(file)[0] + ".txt"
	 	 	 	 	 	 label_file_path = os.path.join(txt_path, txt_file)
 
	 	 	 	 	 	 # 读取标注文件和原始图片
	 	 	 	 	 	 with open(label_file_path, 'r') as f:
	 	 	 	 	 	 	 	 lines = f.readlines()
	 	 	 	 	 	 image = cv2.imread(image_file_path)
 
	 	 	 	 	 	 # 循环处理每个标注框
	 	 	 	 	 	 last_class_id = -1
	 	 	 	 	 	 for line in lines:
	 	 	 	 	 	 	 	 class_id, x_center, y_center, width, height = map(float, line.strip().split())
	 	 	 	 	 	 	 	 if(class_id != last_class_id):
	 	 	 	 	 	 	 	 	 	 n = 1
	 	 	 	 	 	 	 	 	 	 last_class_id = class_id
	 	 	 	 	 	 	 	 else:
	 	 	 	 	 	 	 	 	 	 n = n + 1
	 	 	 	 	 	 	 	 # 将YOLOv5格式的坐标转换为常规坐标
	 	 	 	 	 	 	 	 left = int((x_center - width / 2) * image.shape[1])
	 	 	 	 	 	 	 	 top = int((y_center - height / 2) * image.shape[0])
	 	 	 	 	 	 	 	 right = int((x_center + width / 2) * image.shape[1])
	 	 	 	 	 	 	 	 bottom = int((y_center + height / 2) * image.shape[0])
 
	 	 	 	 	 	 	 	 # 截取标注框内的内容并保存为新图片
	 	 	 	 	 	 	 	 cut_image = image[top:bottom, left:right]
	 	 	 	 	 	 	 	 new_cut_image_path = os.path.join(new_image_path,file)
	 	 	 	 	 	 	 	 new_cut_image_path = new_cut_image_path.replace(".jpg","")
	 	 	 	 	 	 	 	 classid = str(int(class_id))
	 	 	 	 	 	 	 	 new_cut_image_path = new_cut_image_path + "_" + classid + "_" + str(n) + ".jpg"
	 	 	 	 	 	 	 	 print(new_cut_image_path)
	 	 	 	 	 	 	 	 cv2.imwrite(new_cut_image_path, cut_image)
 
if __name__ == "__main__":
	 	 image_path = "./dataset/images"
	 	 txt_path = "./dataset/labels"
	 	 new_image_path = "./dataset/new_images"
	 	 txt_cut_image(image_path,txt_path, new_image_path)
```

然后使用SiameseTool进行分类 ,分隔好以后，大概的目录结构是这样
```
datasets
    └── images_background
        ├── chapter_东
        │   ├── img_3427_0_2.jpg
        │   └── img_3427_1_1.jpg
        ├── chapter_丝
        │   ├── img_3480_0_3.jpg
        │   ├── img_3480_1_3.jpg
        │   ├── img_3544_0_2.jpg
        │   ├── img_3544_1_4.jpg
        │   ├── img_3597_0_3.jpg
        │   ├── img_3597_1_3.jpg
        │   ├── img_3659_0_1.jpg
        │   ├── img_3659_1_2.jpg
        │   ├── img_3697_0_4.jpg
        │   └── img_3697_1_4.jpg
        ├── chapter_为
        │   ├── img_3546_0_1.jpg
        │   ├── img_3546_1_2.jpg
        │   ├── img_3607_0_1.jpg
        │   └── img_3607_1_2.jpg
        ├── chapter_乌
        │   ├── img_3568_0_4.jpg
        │   └── img_3568_1_1.jpg
        ├── chapter_乳
        │   ├── img_3424_0_1.jpg
        │   └── img_3424_1_1.jpg
        ├── chapter_五
        │   ├── img_3460_0_1.jpg
        │   ├── img_3460_1_3.jpg
        │   ├── img_3662_0_1.jpg
        │   └── img_3662_1_3.jpg
        ├── chapter_亭
        │   ├── img_3486_0_3.jpg
        │   ├── img_3486_1_2.jpg
        │   ├── img_3564_0_3.jpg
        │   ├── img_3564_1_2.jpg
        │   ├── img_3627_0_3.jpg
        │   └── img_3627_1_2.jpg
        ├── chapter_仁
        │   ├── img_3402_0_1.jpg
        │   ├── img_3402_1_4.jpg
        │   ├── img_3495_0_1.jpg
        │   ├── img_3495_1_2.jpg
        │   ├── img_3504_0_4.jpg
        │   ├── img_3504_1_2.jpg
        │   ├── img_3514_0_1.jpg
        │   ├── img_3514_1_2.jpg
        │   ├── img_3522_0_1.jpg
        │   ├── img_3522_1_2.jpg
        │   ├── img_3646_0_1.jpg
        │   └── img_3646_1_2.jpg
        ├── chapter_他
        │   ├── img_3474_0_2.jpg
        │   └── img_3474_1_2.jpg

```


下面开始训练，https://github.com/bubbliiiing/siamese-pytorch/
```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```
#### 运行训练时候如果出现 No module named 'torchvision.models.utils' ,就修改 vgg.py第三行
```
import torch
import torch.nn as nn
#from torchvision.models.utils import load_state_dict_from_url 用下面那行代替
from torch.hub import load_state_dict_from_url
```

设置siamese-pytorch\train.py
```
train_own_data = true
save_dir = '' #这个路径就是最后生成权重文件的路径
```

将刚才使用siamese工具生成的训练集拷贝到Siamese-pytorch下（拷贝完成后目录结构：Siamese-pytorch\datasets\images_background）  , train.py，开始训练 .完成后可以使用predict.py


## 三、检测

1.yolo检测图片中所有文字的坐标
当时安装的是这个版本	 yolov5==7.0.13

2.代码实现 (对bilibili的点选验证码进行处理)
```
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

#这个pt文件就是之前yolov5训练好的权重文件
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

```


## playwright

server.py是server服务，针对提交过来的图片进行检测
bilibili.py是测试脚本，模拟登陆bilibli

参考：
https://github.com/MgArcher/Text_select_captcha
https://blog.csdn.net/qq_49268524/article/details/135675169
https://segmentfault.com/a/1190000044549650


