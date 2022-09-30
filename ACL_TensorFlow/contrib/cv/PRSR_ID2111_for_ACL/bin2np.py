import os
import numpy as np
from PIL import Image

def logits_2_pixel_value(logits, mu=1.1):
  """
  Args:
    logits: [n, 256] float32
    mu    : float32
  Returns:
    pixels: [n] float32
  """
  rebalance_logits = logits * mu
  probs = softmax(rebalance_logits)
  pixel_dict = np.arange(0, 256, dtype=np.float32)
  pixels = np.sum(probs * pixel_dict, axis=1)
  return np.floor(pixels)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference

def save_samples(np_imgs, img_path):
  """
  Args:
    np_imgs: [N, H, W, 3] float32
    img_path: str
  """
  np_imgs = np_imgs.astype(np.uint8)
  N, H, W, _ = np_imgs.shape
  num = int(N ** (0.5))
  merge_img = np.zeros((num * H, num * W, 3), dtype=np.uint8)
  for i in range(num):
    for j in range(num):
      merge_img[i*H:(i+1)*H, j*W:(j+1)*W, :] = np_imgs[i*num+j,:,:,:]

  # imsave(img_path, merge_img)
  # misc.imsave(img_path, merge_img)
  #------------------NPU 2021.10.17-------------------------
  img = Image.fromarray(np.uint8(merge_img))
  img.save(img_path)
  #------------------NPU 2021.10.17-------------------------

file_c = "./om_output/2022915_20_33_5_372987/pb_hr_imgs_output_0.bin"
file_p = "./om_output/2022915_20_33_5_372987/pb_hr_imgs_output_1.bin"
out_dir = "./outimg/"

file_line_c = np.fromfile(file_c, dtype='float32')
np_c_logits = file_line_c.reshape((1, 32, 32, 768))
file_line_p = np.fromfile(file_p, dtype='float32')
np_p_logits = file_line_p.reshape((1, 32, 32, 768))

gen_hr_imgs = np.zeros((1, 32, 32, 3), dtype=np.float32)

for i in range(32):
	for j in range(32):
		for c in range(3):
			new_pixel = logits_2_pixel_value(
				np_c_logits[:, i, j, c * 256:(c + 1) * 256] + np_p_logits[:, i, j, c * 256:(c + 1) * 256], mu = 1.0)
			gen_hr_imgs[:, i, j, c] = new_pixel

gen_hr_imgs[0] = gen_hr_imgs[0][:, :, [2,1,0]]
save_samples(gen_hr_imgs, out_dir + '/om_generate' + '.jpg')