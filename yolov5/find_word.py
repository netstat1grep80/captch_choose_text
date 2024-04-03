import torch
from PIL import Image
from pathlib import Path
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

def detect_objects(image_path, model_path='D:\ai\captcha\yolov5\runs\train\exp6\weights\last.pt', conf_thres=0.25, iou_thres=0.45, device=''):
    # 设置设备
    device = select_device(device)

    # 加载模型
    model = attempt_load(model_path, map_location=device)
    model.eval()

    # 加载图片
    img = Image.open(image_path)

    # 图片预处理
    img = letterbox(img, new_shape=model.img_size)[0]

    # 转换为 PyTorch 张量
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)

    # 检测目标
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

    # 提取目标的坐标
    if pred is not None and len(pred) > 0:
        det = pred[:, :4].detach().cpu().numpy()
        return det.tolist()

    return None

# 调用 detect_objects 方法
image_path = 'D:\ai\captcha\yolov5\data\images\bilibili_1.png'
coordinates = detect_objects(image_path)

print(coordinates)
