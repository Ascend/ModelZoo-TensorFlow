# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import PIL
from tqdm import tqdm

def get_random_data(filename_jpg, box, nw, nh):
    """
    修改 box
    :param filename_jpg: 图片名
    :param box: 原box
    :param nw: 改变后的宽度
    :param nh: 改变后的高度
    :return: image, box_resize
    """
    image = Image.open(filename_jpg)
    image = exif_transpose(image)
    iw, ih = image.size
    # 对图像进行缩放并且进行长和宽的扭曲，BICUBIC为三次样条插值，4x4像素临域
    image = image.resize((nw, nh), Image.BICUBIC)
    # 将box进行调整
    box_resize = []
    for boxx in box:
        boxx[0] = str(int(float(boxx[0]) * (nw / iw)))
        boxx[1] = str(int(float(boxx[1]) * (nh / ih)))
        boxx[2] = str(int(float(boxx[2]) * (nw / iw)))
        boxx[3] = str(int(float(boxx[3]) * (nh / ih)))
        box_resize.append(boxx)
    return image, box_resize


def read_xml(xml_name):
    # xml_name: xml文件
    etree = ET.parse(xml_name)
    root = etree.getroot()
    box = []
    for obj in root.iter('object'):
        xmin, ymin, xmax, ymax = (x.text for x in obj.find('bndbox'))
        box.append([xmin, ymin, xmax, ymax])
    return box

def write_xml(xml_name,save_name, box, resize_w, resize_h):
    """
    将修改后的box 写入到 xml文件中
    :param xml_name: 原xml
    :param save_name: 保存的xml
    :param box: 修改后需要写入的box
    """
    etree = ET.parse(xml_name)
    root = etree.getroot()

    # 修改图片的宽度、高度
    for obj in root.iter('size'):
        obj.find('width').text = str(resize_w)
        obj.find('height').text = str(resize_h)

    # 修改box的值
    for obj, bo in zip(root.iter('object'), box):
        for index, x in enumerate(obj.find('bndbox')):
            x.text = bo[index]

    etree.write(save_name)

def start(sourceDir, targetDir, display_dir, w_ratio=1.0, h_ratio=1.0):
    """
    程序开始的主函数
    :param sourceDir: 源文件夹
    :param targetDir: 保存文件夹
    :param resize_w: 改变后的宽度
    :param resize_h: 改变后的高度
    :return:
    """
    assert 0 <= w_ratio <= 1, 'input_params, error: w_ratio between 0-1'
    assert 0 <= h_ratio <= 1, 'input_params, error: h_ratio between 0-1'
    for root, dir1, filenames in os.walk(sourceDir):
        for filename in tqdm(filenames):
            file = os.path.splitext(filename)[0]
            if os.path.splitext(filename)[1] == '.jpg':
                filename_jpg = os.path.join(root, filename)
                image = Image.open(filename_jpg)
                image = exif_transpose(image)
                if image.size[0] > image.size[1]:
                    resize_w, resize_h = 800, 600
                else:
                    resize_w, resize_h = 600, 800
                xml_name = os.path.join('/root/FasterRCNN/VOCdevkit/VOC2012/Annotations/', file + '.xml')
                box = read_xml(xml_name)
                image_data, box_data = get_random_data(filename_jpg, box, resize_w, resize_h)
                # 保存返回的图片
                image_data.save(os.path.join(targetDir, filename))
                # 查看修改后的结果，图片显示
                for j in range(len(box_data)):
                    thickness = 3
                    left, top, right, bottom = box_data[j][0:4]
                    draw = ImageDraw.Draw(image_data)
                    for i in range(thickness):
                        draw.rectangle([int(left) + i, int(top) + i, int(right) - i, int(bottom) - i], outline=(255, 0, 0))
                # 修改xml文件（将修改后的 box 写入到xml文件中）
                save_xml = os.path.join('/root/FasterRCNN/VOCdevkit/VOC2012/Annotations/', file + '.xml')
                write_xml(xml_name, save_xml, box_data, resize_w, resize_h)

# 解决PIL读取图片时自动旋转的问题
def exif_transpose(img):
    if not img:
        return img
    exif_orientation_tag = 274
    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]
        # Handle EXIF Orientation
        if orientation == 1:  # Normal image - nothing to do!
            pass
        elif orientation == 2:  # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:  # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:  # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:  # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:  # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:  # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:  # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img

if __name__ == "__main__":
    print('start....')
    sourceDir = "/root/FasterRCNN/VOCdevkit/VOC2012/JPEGImages" # 源文件夹；需要将jpg文件和xml文件放在同一个文件夹；
    targetDir = "/root/FasterRCNN/VOCdevkit/VOC2012/JPEGImages"    # 目标文件夹；生成新的jpg和xml；
    display_dir = "/root/FasterRCNN/temp"# 显示效果文件夹；
    w_ratio, h_ratio = 0.25, 0.25  # 宽高缩放比例，如400*0.25=100
    print('processing')
    start(sourceDir, targetDir, display_dir, w_ratio, h_ratio)
    print('done!')
