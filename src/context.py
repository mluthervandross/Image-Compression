# -*- coding: utf-8 -*-
# Copyright 2019 Google LLC. All Rights Reserved.
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
# ==============================================================================
"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
import math
import logging
from logging import handlers
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc

#from signal_conv_ import SignalConv2D
import arithmeticcoding
import os


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

a_mse = []
a_psnr = []
a_msssim = []
a_msssim_dB = []
a_eval_bpp = []
a_bpp = []

def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def quantize(tensor, training=False, mean=None):
  """Add noise or quantize"""
  half = tf.constant(.5, dtype=tf.float32)
  if training:
    noise = tf.random.uniform(tf.shape(tensor), -half, half)
    tensor = tf.math.add_n([tensor, noise])
  else:
    if mean is not None:
      tensor -= mean
      tensor = tf.math.floor(tensor + half)
      tensor += mean
    else:
      tensor = tf.math.floor(tensor + half)
    # tensor = tf.round(tensor)
  return tensor


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)
  
  
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'crit':logging.CRITICAL
    }
    def __init__(self, filename, level='info', when='D', backCount=3,
            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_2")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(AnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class SynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(SynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_2", inverse=True)),
        tfc.SignalConv2D(
            3, (5, 5), name="layer_3", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(SynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=False,
            activation=None),
    ]
    super(HyperAnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            288, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            384, (3, 3), name="layer_2", corr=False, strides_up=1,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=None),
    ]
    super(HyperSynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class MaskedConvolution2D(tf.keras.layers.Layer):
    """
    This model generate context of the latent.
    'a' for google's context;
    'b' for another context of us
    others for our context in VCIP'2020
    """

    def __init__(self, *args, **kwargs):
      super(MaskedConvolution2D, self).__init__(*args, **kwargs)

    def build(self, input_shape):
      self._layers = [
          tfc.SignalConv2D(
            384, (5, 5), name="layer_0", mask='a', corr=False, strides_up=1,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None, 
            activation=None)
      ]
      super(MaskedConvolution2D, self).build(input_shape)

    def call(self, tensor):
      for layer in self._layers:
          tensor = layer(tensor)
      return tensor


class EntropyParameters(tf.keras.layers.Layer):
    """The prediction of mu and sigma for entropy estimation."""

    def __init__(self, *args, **kwargs):
        super(EntropyParameters, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            SignalConv2D(
              640, (1, 1), name="layer_0", corr=False, strides_up=1,
              padding="same_zeros", use_bias=True, kernel_parameterizer=None, 
              activation=tf.nn.leaky_relu), 
            SignalConv2D(
              512, (1, 1), name="layer_1", corr=False, strides_up=1,
              padding="same_zeros", use_bias=True, kernel_parameterizer=None, 
              activation=tf.nn.leaky_relu), 
            SignalConv2D(
              384, (1, 1), name="layer_2", corr=False, strides_up=1,
              padding="same_zeros", use_bias=True, kernel_parameterizer=None, 
              activation=None)
        ]
        super(EntropyParameters, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        mu, sigma = tf.split(tensor, [192, 192], axis=3)
        return mu, sigma


def train(args):
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device("/cpu:0"):
    train_files = glob.glob(args.train_glob)
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  synthesis_transform = SynthesisTransform(args.num_filters)
  hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
  hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  context_prediction = MaskedConvolution2D()
  entropy_prediction = EntropyParameters()

  # Build autoencoder and hyperprior.
  y = analysis_transform(x)
  z = hyper_analysis_transform(abs(y))
  z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
  sigma_ = hyper_synthesis_transform(z_tilde)
  #***************************************************************************        
  # Initial MaskedConvolution2D
  # y_bar = quantize(y, training=True)  #Add uniform noise
  # context = MaskedConvolution2D(y_bar, (5, 5, 384)).output()
  context = context_prediction(y)
  prediction = tf.concat([context, sigma_], axis=3)
  mean, sigma = entropy_prediction(prediction)
  #***************************************************************************
  # mean, sigma = tf.split(pred, [192, 192], axis=3)

  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, mean=mean)
  y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde)

  # Total number of bits divided by number of pixels.
  train_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
               tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)
  # train_bpp = (tf.reduce_sum(tf.log(y_likelihoods))) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2

  # The rate-distortion cost.
  train_loss = args.lmbda * train_mse + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
  # train_op = tf.group(main_step)

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse)

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]
  iter = 0
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_secs=60) as sess:
    while not sess.should_stop():
        _, train_loss_ = sess.run([train_op, train_loss])
        if iter % 1000 == 0:
          # print("Step {}: Loss {:0.8f}".format(iter + 1, train_loss_))
          log.logger.info("Step {}: loss {:0.8f}".format(iter + 1, train_loss_))
        iter  += 1
        #sess.run(train_op)


def compress(args):
  """Compresses an image."""

  with tf.device("/cpu:0"):
    test_files = glob.glob(args.compress_glob)
    if not test_files:
      raise RuntimeError(
            "No test images found with glob '{}'.".format(args.compress_glob))

  for input_file in test_files:
      file = input_file.split("/")
      file = file[-1]
      file = file.split(".")
      output_file = file[-2]
      output_file = "./results/lmbda_0.1_1x1/test/" + output_file + ".tfci"
      
      tf.reset_default_graph()

      # Load input image and add batch dimension.
      x = read_png(input_file)
      x = tf.expand_dims(x, 0)
      x.set_shape([1, None, None, 3])
      x_shape = tf.shape(x)

      # Instantiate model.
      analysis_transform = AnalysisTransform(args.num_filters)
      synthesis_transform = SynthesisTransform(args.num_filters)
      hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
      hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
      entropy_bottleneck = tfc.EntropyBottleneck()
      context_prediction = MaskedConvolution2D()
      entropy_prediction = EntropyParameters()

      # Transform and compress the image.
      y = analysis_transform(x)
      y_shape = tf.shape(y)
      z = hyper_analysis_transform(abs(y))
      z_hat, z_likelihoods = entropy_bottleneck(z, training=False)
      sigma_ = hyper_synthesis_transform(z_hat)
      #***************************************************************************
      # Instantiate context based rate prediction model
      y_bar = quantize(y)
      context = context_prediction(y_bar)
      prediction = tf.concat([context, sigma_], axis=3)
      mean, sigma = entropy_prediction(prediction)
      #***************************************************************************
      mean = mean[:, :y_shape[1], :y_shape[2], :]
      sigma = sigma[:, :y_shape[1], :y_shape[2], :]
      scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
      conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, dtype=tf.float32, mean=mean)
      side_string = entropy_bottleneck.compress(z)

      # Transform the quantized image back (if requested).
      y_hat, y_likelihoods = conditional_bottleneck(y, training=False)
      # y_hat = conditional_bottleneck._quantize(y, mode='symbols')
      string = conditional_bottleneck.compress(y)
      x_hat = synthesis_transform(y_bar)
      x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

      num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

      # Total number of bits divided by number of pixels.
      eval_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) \
                    + tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

      # Bring both images back to 0..255 range.
      x *= 255
      x_hat = tf.clip_by_value(x_hat, 0, 1)
      x_hat = tf.round(x_hat * 255)

      mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
      psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
      msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

      with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)

        tensors = [string, side_string, tf.shape(x)[1:-1], tf.shape(y)[1:-1], tf.shape(z)[1:-1]]
#         tensors = [side_string, tf.shape(z)[1:-1]]
         arrays = sess.run(tensors)

         # Write a binary file with the shape information and the compressed string.
         packed = tfc.PackedTensors()
         packed.pack(tensors, arrays)
         with open(output_file, "wb") as f:
           f.write(packed.string)

#        y_hat_, mean_, sigma_, x_shape_ = sess.run([y_bar, mean, sigma, tf.shape(x)[1:-1]])
#
#        fileobj = open(compressed_file_path, mode='wb')
#        arr = np.array([x_shape_[0], x_shape_[1]], dtype=np.uint16)
#        arr.tofile(fileobj)
#        fileobj.close()
#        bitout = arithmeticcoding.BitOutputStream(open(compressed_file_path, 'ab+'))
#        enc = arithmeticcoding.ArithmeticEncoder(bitout)
#        for ch_idx in range(y_hat_.shape[-1]):
#          for h_idx in range(y_hat_.shape[1]):
#            for w_idx in range(y_hat_.shape[2]):
#              mu_val = mean_[0, h_idx, w_idx, ch_idx]
#              sigma_val = abs(sigma_[0, h_idx, w_idx, ch_idx])
#              freq = arithmeticcoding.ModelFrequencyTable(mu_val + 255, sigma_val)
#              print(y_hat_[0, h_idx, w_idx, ch_idx])
#              symbol = y_hat_[0, h_idx, w_idx, ch_idx] + 255
#              if symbol < 0 or symbol > 511:
#                print("symbol range error: " + str(symbol))
#              enc.write(freq, symbol)
#        enc.write(freq, 512)
#        enc.finish()
#        bitout.close()

        # If requested, transform the quantized image back and measure performance.
        if args.verbose:
          eval_bpp, mse, psnr, msssim, num_pixels = sess.run(
              [eval_bpp, mse, psnr, msssim, num_pixels])

          # The actual bits per pixel including overhead.
          bpp = len(packed.string) * 8 / num_pixels
#          bpp = (len(packed.string) + os.path.getsize(compressed_file_path)) * 8 / num_pixels

          a_mse.append(mse)
          a_msssim.append(msssim)
          a_psnr.append(psnr)
          a_msssim_dB.append(-10 * np.log10(1 - msssim))
          a_eval_bpp.append(eval_bpp)
          a_bpp.append(bpp)

          log.logger.info("Image {}: mse {:0.8f}   psnr {:0.8f}   msssim {:0.8f}  dB {:0.8f}   eval_bpp {:0.8f}   bpp {:0.8f}"
                          .format(input_file, mse, psnr, msssim, -10 * np.log10(1 - msssim), eval_bpp, bpp))

  log.logger.info("Average performance: mse {:0.8f}   psnr {:0.8f}   msssim {:0.8f}    dB {:0.8f}   eval_bpp {:0.8f}   bpp {:0.8f}"
                  .format(np.mean(a_mse), np.mean(a_psnr), np.mean(a_msssim), np.mean(a_msssim_dB), np.mean(a_eval_bpp), np.mean(a_bpp)))


def decompress(args):
  """Decompresses an image."""

  with tf.device("/cpu:0"):
      binary_files = glob.glob(args.decompress_glob)
      if not binary_files:
        raise RuntimeError(
              "No test images found with glob '{}'.".format(args.decompress_glob))

  for input_file in binary_files:
      file = input_file.split("/")
      file = file[-1]
      file = file.split(".")
      output_file = file[-2]
      output_file = "./results/lmbda_0.01_1x1/test/" + output_file + ".png"

      tf.reset_default_graph()

      # Read the shape information and compressed string from the binary file.
      string = tf.placeholder(tf.string, [1])
      side_string = tf.placeholder(tf.string, [1])
      x_shape = tf.placeholder(tf.int32, [2])
      y_shape = tf.placeholder(tf.int32, [2])
      z_shape = tf.placeholder(tf.int32, [2])
      with open(input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
      tensors = [string, side_string, x_shape, y_shape, z_shape]
      arrays = packed.unpack(tensors)
      string, side_string, x_shape, y_shape, z_shape = arrays
      side_string, z_shape = arrays
      z_shape_ = z_shape

      # Instantiate model.
      synthesis_transform = SynthesisTransform(args.num_filters)
      hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
      entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
      entropy_prediction = EntropyParameters()
      context_prediction = MaskedConvolution2D()

      # Decompress and transform the image back.
      z_shape = tf.concat([z_shape, [args.num_filters]], axis=0)
      z_hat = entropy_bottleneck.decompress(side_string, z_shape, channels=args.num_filters)
      sigma_ = hyper_synthesis_transform(z_hat)
      #*************************************************************************************************
      y_bar = tf.placeholder(tf.float32, shape=(1, y_shape[0], y_shape[1], args.num_filters))
      # y_bar = np.zeros((1, y_shape_[0], y_shape_[1], args.num_filters), dtype=np.int32)
      # context = MaskedConvolution2D(y_bar, (5, 5, 384)).output()
      context = context_prediction(y_bar)
      prediction = tf.concat([context, sigma_], axis=3)
      mean, sigma = entropy_prediction(prediction)
      # sigma = EntropyPrediction(prediction)
      #*************************************************************************************************
      mean = mean[:, :y_shape[0], :y_shape[1], :]
      sigma = sigma[:, :y_shape[0], :y_shape[1], :]
      scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
      conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, dtype=tf.float32, mean=mean)
      # conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, dtype=tf.float32)
      y_hat = conditional_bottleneck.decompress(string)
      x_hat = synthesis_transform(tf.round(y_bar))

      # Remove batch dimension, and crop away any extraneous padding on the bottom or right boundaries.
      x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

      # Write reconstructed image out as a PNG file.
      op = write_png(output_file, x_hat)

      # Load the latest model checkpoint, and perform the above actions.
      with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)

        y_bar_ = np.zeros((1, y_shape[0], y_shape[1], args.num_filters), dtype=np.float32)
        for h_idx in range(y_shape[0]):
            for w_idx in range(y_shape[1]):
               y_hat_, mean_ = sess.run([y_hat, mean], feed_dict={y_bar:np.around(y_bar_)})
               y_bar_[:, h_idx, w_idx, :] = y_hat_[:, h_idx, w_idx, :] - mean_[:, h_idx, w_idx, :]
        
        sess.run(op, feed_dict={y_bar: y_bar_})


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="./models/mse/lmbda_0.1_1x1",
      help="Directory where to save/load model checkpoints.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model.")
  train_cmd.add_argument(
      "--train_glob", default="../data/mobile/train/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")
  compress_cmd.add_argument(
      "--compress_glob", default="../../data/test/*.png", 
      help="Glob pattern identifying testing data. This pattern must expand to "
           "a list of RGB images in PNG format.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")
  decompress_cmd.add_argument(
      "--decompress_glob", default="./results/lmbda_0.01_1x1/test/*.tfci", 
      help="Glob pattern identifying binary data. This pattern must expand to "
            "a list of binary files in tfci format.")

  # # Arguments for both 'compress' and 'decompress'.
  # for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
  #   cmd.add_argument(
  #       "input_file",
  #       help="Input filename.")
  #   cmd.add_argument(
  #       "output_file", nargs="?",
  #       help="Output filename (optional). If not provided, appends '{}' to "
  #            "the input filename.".format(ext))

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    # if not args.output_file:
    #   args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "decompress":
    # if not args.output_file:
    #   args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  log = Logger('performance.log', level='info')
  app.run(main, flags_parser=parse_args)
