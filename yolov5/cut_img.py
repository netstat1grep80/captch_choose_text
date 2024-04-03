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