import errno
import numpy as np
import cv2
import os


def read_image(path):
  # bgr image
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(img, path)
  img = img.astype(np.float32)
  img = preprocess(img)
  return img

def write_image(path, img):
  img = postprocess(img)
  cv2.imwrite(path, img)

def preprocess(img):
  imgpre = np.copy(img)
  imgpre = imgpre[...,::-1]
  imgpre = imgpre[np.newaxis,:,:,:]
  imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  return imgpre

def postprocess(img):
  imgpost = np.copy(img)
  imgpost += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  imgpost = imgpost[0]
  imgpost = np.clip(imgpost, 0, 255).astype('uint8')
  imgpost = imgpost[...,::-1]
  return imgpost

def normalize(weights):
  denom = sum(weights)
  if denom > 0.:
    return [float(i) / denom for i in weights]
  else: return [0.] * len(weights)

def mkdir(dir_path):
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def check_image(img, path):
  if img is None:
    raise OSError(errno.ENOENT, "No such file", path)
