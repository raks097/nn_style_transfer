import tensorflow as tf
import numpy as np
import argparse
import time
import cv2
import os

from model import build_model
from util import write_image,mkdir,normalize,check_image,preprocess

'''
  parsing and configuration
'''
def parse_arguments():

  parser = argparse.ArgumentParser()

  # single image options
  parser.add_argument('--image_name', type=str, default='result', help='output image file name.')
  parser.add_argument('--style_imgs', nargs='+', type=str, help='style images filenames', required=True)
  parser.add_argument('--style_imgs_weights', nargs='+', type=float,default=[1.0], help='Interpolation weights of each of the style images. (example: 0.5 0.5)')
  parser.add_argument('--content_img', type=str, help='Filename of the content image (example: lion.jpg)')
  parser.add_argument('--style_imgs_dir', type=str, default='./styles', help='Directory path to the style images. (default: %(default)s)')
  parser.add_argument('--content_img_dir', type=str, default='./image_input', help='Directory path to the content image. (default: %(default)s)')
  parser.add_argument('--max_size', type=int, default=512, help='Maximum width or height of the input images. (default: %(default)s)')
  parser.add_argument('--content_weight', type=float, default=5e0, help='Weight for the content loss function. (default: %(default)s)')
  parser.add_argument('--style_weight', type=float, default=1e4, help='Weight for the style loss function. (default: %(default)s)')
  parser.add_argument('--tv_weight', type=float, default=1e-3, help='Weight for the total variational loss function. Set small (e.g. 1e-3). (default: %(default)s)')
  parser.add_argument('--content_layers', nargs='+', type=str, default=['conv4_2'], help='VGG19 layers used for the content image. (default: %(default)s)')
  parser.add_argument('--style_layers', nargs='+', type=str, default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], help='VGG19 layers used for the style image. (default: %(default)s)')
  parser.add_argument('--content_layer_weights', nargs='+', type=float, default=[1.0], help='Contributions (weights) of each content layer to loss. (default: %(default)s)')
  parser.add_argument('--style_layer_weights', nargs='+', type=float, default=[0.2, 0.2, 0.2, 0.2, 0.2], help='Contributions (weights) of each style layer to loss. (default: %(default)s)')
  parser.add_argument('--style_mask', action='store_true', help='Transfer the style to masked regions.')
  parser.add_argument('--style_mask_imgs', nargs='+', type=str, default=None, help='Filenames of the style mask images (example: face_mask.png) (default: %(default)s)')
  # the above is the dataset.
  parser.add_argument('--model_weights', type=str, default='imagenet-vgg-verydeep-19.mat', help='Weights and biases of the VGG-19 network.')
  parser.add_argument('--pooling_type', type=str, default='avg', choices=['avg', 'max'], help='Type of pooling in convolutional neural network. (default: %(default)s)')
  parser.add_argument('--device', type=str, default='/gpu:0', choices=['/gpu:0', '/cpu:0'], help='GPU or CPU mode.  GPU mode requires NVIDIA CUDA. (default|recommended: %(default)s)')
  parser.add_argument('--img_output_dir', type=str, default='./image_output', help='Relative or absolute directory path to output image and data.')

  # optimization
  parser.add_argument('--optimizer', type=str, default='lbfgs', choices=['lbfgs', 'adam'], help='Loss minimization optimizer.  L-BFGS gives better results.  Adam uses less memory. (default|recommended: %(default)s)')
  parser.add_argument('--learning_rate', type=float, default=1e0, help='Learning rate parameter for the Adam optimizer. (default: %(default)s)')
  parser.add_argument('--max_iterations', type=int, default=50, help='Max number of iterations for the Adam or L-BFGS optimizer. (default: %(default)s)')

  args = parser.parse_arguments()

  # normalize weights
  args.style_layer_weights   = normalize(args.style_layer_weights)
  args.content_layer_weights = normalize(args.content_layer_weights)
  args.style_imgs_weights    = normalize(args.style_imgs_weights)

  # create directories for output
  mkdir(args.img_output_dir)

  return args

#STYLING FUNCTIONS

def content_layer_loss_func(p, x):
  _, h, w, d = p.get_shape()
  m = h.value * w.value
  n = d.value
  k = 1. / (2. * n**0.5 * m**0.5)
  loss = k * tf.reduce_sum(tf.pow((x - p), 2))
  return loss

def style_layer_loss_func(a, x):
  _, h, w, d = a.get_shape()
  m = h.value * w.value
  n = d.value
  A = gram_matrix(a, m, n)
  G = gram_matrix(x, m, n)
  loss = (1./(4 * n**2 * m**2)) * tf.reduce_sum(tf.pow((G - A), 2))
  return loss

def gram_matrix(x, area, depth):
  F = tf.reshape(x, (area, depth))
  G = tf.matmul(tf.transpose(F), F)
  return G

def mask_style_layer(a, x, mask_image):
  _, h, w, d = a.get_shape()
  mask = get_mask_image(mask_image, w.value, h.value)
  mask = tf.convert_to_tensor(mask)
  tensors = []
  for _ in range(d.value):
    tensors.append(mask)
  mask = tf.stack(tensors, axis=2)
  mask = tf.stack(mask, axis=0)
  mask = tf.expand_dims(mask, 0)
  a = tf.multiply(a, mask)
  x = tf.multiply(x, mask)
  return a, x

def sum_masked_style_losses(sess, net, style_imgs):
  total_style_loss = 0.
  weights = args.style_imgs_weights
  masks = args.style_mask_imgs
  for img, img_weight, img_mask in zip(style_imgs, weights, masks):
    sess.run(net['input'].assign(img))
    style_loss = 0.
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
      a = sess.run(net[layer])
      x = net[layer]
      a = tf.convert_to_tensor(a)
      a, x = mask_style_layer(a, x, img_mask)
      style_loss += style_layer_loss_func(a, x) * weight
    style_loss /= float(len(args.style_layers))
    total_style_loss += (style_loss * img_weight)
  total_style_loss /= float(len(style_imgs))
  return total_style_loss

def sum_style_losses(sess, net, style_imgs):
  total_style_loss = 0.
  weights = args.style_imgs_weights
  for img, img_weight in zip(style_imgs, weights):
    sess.run(net['input'].assign(img))
    style_loss = 0.
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
      a = sess.run(net[layer])
      x = net[layer]
      a = tf.convert_to_tensor(a)
      style_loss += style_layer_loss_func(a, x) * weight
    style_loss /= float(len(args.style_layers))
    total_style_loss += (style_loss * img_weight)
  total_style_loss /= float(len(style_imgs))
  return total_style_loss

def sum_content_losses(sess, net, content_img):
  sess.run(net['input'].assign(content_img))
  content_loss = 0.
  for layer, weight in zip(args.content_layers, args.content_layer_weights):
    p = sess.run(net[layer])
    x = net[layer]
    p = tf.convert_to_tensor(p)
    content_loss += content_layer_loss_func(p, x) * weight
  content_loss /= float(len(args.content_layers))
  return content_loss




# Ouput

def stylize(content_img, style_imgs, init_img, frame=None):
  with tf.device(args.device), tf.Session() as sess:
    net = build_model(content_img,args.model_weights,args.pooling_type)
    if args.style_mask:
      L_style = sum_masked_style_losses(sess, net, style_imgs)
    else:
      L_style = sum_style_losses(sess, net, style_imgs)
    L_content = sum_content_losses(sess, net, content_img)
    L_tv = tf.image.total_variation(net['input'])
    alpha = args.content_weight
    beta  = args.style_weight
    theta = args.tv_weight
    L_total  = alpha * L_content
    L_total += beta  * L_style
    L_total += theta * L_tv
    optimizer = get_optimizer(L_total)
    if args.optimizer == 'adam':
      minimize_with_adam(sess, net, optimizer, init_img, L_total)
    elif args.optimizer == 'lbfgs':621
      minimize_with_lbfgs(sess, net, optimizer, init_img)
    output_img = sess.run(net['input'])
    write_image_output(output_img, content_img, style_imgs, init_img)

def minimize_with_lbfgs(sess, net, optimizer, init_img):
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  sess.run(net['input'].assign(init_img))
  optimizer.minimize(sess)

def minimize_with_adam(sess, net, optimizer, init_img, loss):
  train_op = optimizer.minimize(loss)
  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  sess.run(net['input'].assign(init_img))
  iterations = 0
  while (iterations < args.max_iterations):
    sess.run(train_op)
    iterations += 1

def get_optimizer(loss):
  if args.optimizer == 'lbfgs':
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',options={'maxiter': args.max_iterations})
  elif args.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
  return optimizer

# Image Loading

def get_init_image(content_img, style_imgs):
  return content_img


def get_content_image(content_img):
  path = os.path.join(args.content_img_dir, content_img)
   # bgr image
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(img, path)
  img = img.astype(np.float32)
  h, w, d = img.shape
  mx = args.max_size
  # resize if > max size
  if h > w and h > mx:
    w = (float(mx) / float(h)) * w
    img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
  if w > mx:
    h = (float(mx) / float(w)) * h
    img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
  img = preprocess(img)
  return img

def get_style_images(content_img):
  _, ch, cw, cd = content_img.shape
  style_imgs = []
  for style_fn in args.style_imgs:
    path = os.path.join(args.style_imgs_dir, style_fn)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    style_imgs.append(img)
  return style_imgs


def get_mask_image(mask_img, width, height):
  path = os.path.join(args.content_img_dir, mask_img)
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  check_image(img, path)
  img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
  img = img.astype(np.float32)
  mx = np.amax(img)
  img /= mx
  return img

def write_image_output(output_img, content_img, style_imgs, init_img):
  out_dir = os.path.join(args.img_output_dir, args.image_name)
  mkdir(out_dir)
  img_path = os.path.join(out_dir, args.image_name+'.png')
  content_path = os.path.join(out_dir, 'content.png')
  init_path = os.path.join(out_dir, 'init.png')

  write_image(img_path, output_img)
  write_image(content_path, content_img)
  write_image(init_path, init_img)
  index = 0
  for style_img in style_imgs:
    path = os.path.join(out_dir, 'style_'+str(index)+'.png')
    write_image(path, style_img)
    index += 1


def render_single_image():
  content_img = get_content_image(args.content_img)
  style_imgs = get_style_images(content_img)
  with tf.Graph().as_default():
    print('\n---- RENDERING SINGLE IMAGE ----\n')
    init_img = get_init_image(content_img, style_imgs)
    tick = time.time()
    stylize(content_img, style_imgs, init_img)
    tock = time.time()
    print('Single image elapsed time: {}'.format(tock - tick))


def main():
  global args
  args = parse_arguments()
  render_single_image()

if __name__ == '__main__':
  main()
