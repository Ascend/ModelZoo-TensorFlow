import os
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
lfw_path="/home/zhang/dataset/lfw-deepfunneled_align"
output_path="/home/zhang/dataset/lfw_bin"

def load_and_preprocess_image(img_path):
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.cond(
        tf.image.is_jpeg(img_raw),
        lambda: tf.image.decode_jpeg(img_raw,3),
        lambda: tf.image.decode_png(img_raw,3))
    img_tensor = tf.image.resize(img_tensor, [112, 112])
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = (img_tensor - 127.5) / 128.0
    return img_tensor


print("lfw_path: ",lfw_path)
print("output_path: ",output_path)
os.makedirs(output_path,exist_ok=True)
print("convert image to bin......")

for cls in tqdm(os.listdir(lfw_path)):
    cls_path=os.path.join(lfw_path,cls)
    if os.path.isdir(cls_path):
        for file in os.listdir(cls_path):
            file_path=os.path.join(cls_path,file)
            img=load_and_preprocess_image(file_path)
            img=img.numpy()

            outFile_path=os.path.join(output_path,file[:-4]+".bin")
            img.tofile(outFile_path)

print("finish convert!")