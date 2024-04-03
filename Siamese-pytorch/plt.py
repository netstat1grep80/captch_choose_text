
from skimage import data,color
from PIL import Image
import yolov5


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def get_image_from_coordinates(image, coordinates):
    # 假设 coordinates 是一个四元组 (x1, y1, x2, y2)，表示左上角和右下角的坐标
    x1, y1, x2, y2 = coordinates
    # 假设 image 是包含所有图像的一个列表
    # 你需要根据实际情况修改这里的代码，确保从 image 中正确截取出对应坐标的图像
    return image.crop((x1, y1, x2, y2))


def show_plt():
    img = Image.open('bilibili.png')


    fig, axes = plt.subplots(2, 4, figsize=(15, 6))

    img1 = get_image_from_coordinates(img, [384.2741394042969, 2.6555275917053223, 437.2212219238281, 79.66405487060547])

    for i in range(4):
        axes[0,i].imshow(img1)
        axes[0,i].set_title("Original image 0,{}".format(i))
        axes[0,i].axis('off')

        axes[1,i].imshow(img1)
        axes[1,i].set_title("Original image 1,{}".format(i))
        axes[1,i].axis('off')
   
    fig.tight_layout()  #自动调整subplot间的参数
    print(type(plt))
    plt.show()

show_plt()