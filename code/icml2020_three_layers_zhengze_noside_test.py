import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import tensorflow.keras.backend as K
from octave import OctConv2D, OctConv2DTranspose, GoConv, GoTConv
from PIL import Image
import os
from evaluate import evaluate
from tensorflow_compression.python.layers import parameterizers

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

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


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(Encoder, self).__init__(*args, **kwargs)

  def build(self, input_shape):    
    super(Encoder, self).build(input_shape)

  def call(self, tensor):
    high, low = GoConv(self.num_filters, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_0", corr=True, use_bias=True, activation=tfc.GDN(name="gdn_0"))([tensor])
    high, low = GoConv(self.num_filters, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_1", corr=True, use_bias=True, activation=tfc.GDN(name="gdn_1"))([high, low])
    high, low = GoConv(self.num_filters, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_2", corr=True, use_bias=True, activation=tfc.GDN(name="gdn_2"))([high, low])
    high, low = GoConv(self.num_filters, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_3", corr=True, use_bias=True, activation=None)([high, low])
    return high, low

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(Decoder, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    super(Decoder, self).build(input_shape)

  def call(self, tensor):    
    high, low = tensor
    high, low = GoTConv(self.num_filters, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_0", corr=False, use_bias=True, activation=tfc.GDN(name="igdn_0", inverse=True))([high, low])
    high, low = GoTConv(self.num_filters, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_1", corr=False, use_bias=True, activation=tfc.GDN(name="igdn_1", inverse=True))([high, low])
    high, low = GoTConv(self.num_filters, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_2", corr=False, use_bias=True, activation=tfc.GDN(name="igdn_2", inverse=True))([high, low])
    high = GoTConv(3, 0.0, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_3", corr=False, use_bias=True, activation=None)([high, low])
    #high, low = GoTConv(self.num_filters//32, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_3", corr=False, use_bias=True, activation=None)([high, low])
    #return high, low
    return high

class HyperEncoder(tf.keras.layers.Layer):
  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperEncoder, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    super(HyperEncoder, self).build(input_shape)

  def call(self, tensor):
    high, low = tensor
    high, low = GoConv(self.num_filters, 0.5, kernel_size=(3,3), padding="same_zeros", strides=1, name="layer_0", corr=True, use_bias=True, activation=tf.nn.leaky_relu)([high, low])
    high, low = GoConv(self.num_filters, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_1", corr=True, use_bias=True, activation=tf.nn.leaky_relu)([high, low])
    high, low = GoConv(self.num_filters, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_2", corr=True, use_bias=False, activation=None)([high, low])
    return high, low
    

class HyperDecoder(tf.keras.layers.Layer):
  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperDecoder, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    super(HyperDecoder, self).build(input_shape)

  def call(self, tensor):
    high, low = tensor
    high, low = GoTConv(self.num_filters, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_0", corr=False, use_bias=True, kernel_parameterizer=None, activation=tf.nn.leaky_relu)([high, low])
    high, low = GoTConv(288, 0.5, kernel_size=(5,5), padding="same_zeros", strides=2, name="layer_1", corr=False, use_bias=True, kernel_parameterizer=None, activation=tf.nn.leaky_relu)([high, low])
    high, low = GoTConv(self.num_filters * 2, 0.5, kernel_size=(3,3), padding="same_zeros", strides=1, name="layer_2", corr=False, use_bias=True, kernel_parameterizer=None, activation=None)([high, low])        
    return high, low

class HyperEncoder_second(tf.keras.layers.Layer):
  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperEncoder_second, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    super(HyperEncoder_second, self).build(input_shape)

  def call(self, tensor):
    high, low = tensor
    high, low = GoConv(self.num_filters, 0.5, kernel_size=(3,3), padding="same_zeros", strides=1, name="layer_0", corr=True, use_bias=True, activation=tf.nn.leaky_relu)([high, low])
    high, low = GoConv(self.num_filters, 0.5, kernel_size=(3,3), padding="same_zeros", strides=1, name="layer_1", corr=True, use_bias=True, activation=tf.nn.leaky_relu)([high, low])
    high, low = GoConv(self.num_filters, 0.5, kernel_size=(3,3), padding="same_zeros", strides=1, name="layer_2", corr=True, use_bias=False, activation=None)([high, low])
    return high, low

class HyperDecoder_second(tf.keras.layers.Layer):
  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperDecoder_second, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    super(HyperDecoder_second, self).build(input_shape)

  def call(self, tensor):
    high, low = tensor
    high, low = GoTConv(self.num_filters, 0.5, kernel_size=(3,3), padding="same_zeros", strides=1, name="layer_0", corr=False, use_bias=True, kernel_parameterizer=None, activation=tf.nn.leaky_relu)([high, low])
    high, low = GoTConv(288, 0.5, kernel_size=(3,3), padding="same_zeros", strides=1, name="layer_1", corr=False, use_bias=True, kernel_parameterizer=None, activation=tf.nn.leaky_relu)([high, low])
    high, low = GoTConv(self.num_filters * 2, 0.5, kernel_size=(3,3), padding="same_zeros", strides=1, name="layer_2", corr=False, use_bias=True, kernel_parameterizer=None, activation=None)([high, low])        
    return high, low	
	
	
class EntropyParam(tf.keras.layers.Layer):
  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(EntropyParam, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(640, (1, 1), name="layer_0", corr=True, strides_down=1, padding="same_zeros", use_bias=True,activation=tf.nn.leaky_relu),
        tfc.SignalConv2D(512, (1, 1), name="layer_1", corr=True, strides_down=1, padding="same_zeros", use_bias=True,activation=tf.nn.leaky_relu),
        tfc.SignalConv2D(self.num_filters * 2, (1, 1), name="layer_2", corr=True, strides_down=1, padding="same_zeros", use_bias=False,activation=None),
    ]
    super(EntropyParam, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor[:,:,:,:self.num_filters], tensor[:,:,:,self.num_filters:]
	
	
	
class EntropyParam_second(tf.keras.layers.Layer):
  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(EntropyParam_second, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(640, (1, 1), name="layer_0", corr=True, strides_down=1, padding="same_zeros", use_bias=True,activation=tf.nn.leaky_relu),
        tfc.SignalConv2D(512, (1, 1), name="layer_1", corr=True, strides_down=1, padding="same_zeros", use_bias=True,activation=tf.nn.leaky_relu),
        tfc.SignalConv2D(self.num_filters * 2, (1, 1), name="layer_2", corr=True, strides_down=1, padding="same_zeros", use_bias=False,activation=None),
    ]
    super(EntropyParam_second, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor[:,:,:,:self.num_filters], tensor[:,:,:,self.num_filters:]

def maskedConv2d(inputs, input_channels=96, kernel_size=5, stride=1, padding="SAME", scope="conv2d"):
  #weights_initializer = tf.contrib.layers.xavier_initializer()
  weights_initializer = tf.uniform_unit_scaling_initializer()
  with tf.variable_scope(scope):
    center = kernel_size // 2
    weights_shape = [kernel_size, kernel_size, input_channels, input_channels*2]
    weights = tf.get_variable("weights", weights_shape, tf.float32, weights_initializer)
    mask = np.ones((kernel_size, kernel_size, input_channels, input_channels*2), dtype=np.float32)
    mask[center, center+1: ,: ,:] = 0.
    mask[center+1:, :, :, :] = 0.      
    # if mask_type = 'A'
    mask[center,center,:,:] = 0.
    weights = weights * tf.constant(mask, dtype=tf.float32)
    outputs = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1], padding=padding, name='outputs')
    return outputs
    

class Up_layers(tf.keras.layers.Layer): ###old is num_filters//8
  def __init__(self, num_filters=256):
    super(Up_layers, self).__init__()
    self.hyper_low_up_high = tfc.SignalConv2D(num_filters//2, (3,3), name="hyper_low_up_high", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="hyper_low_up_high"))
    
    self.decode_low_up_high = tfc.SignalConv2D(num_filters//2, (3,3), name="decode_low_up_high", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="decode_low_up_high"))
    

 
  def call(self, x):
    x = self.hyper_low_up_high(x) ### put the hyper_low to become the same dimension  of the hyper_high
    x = self.decode_low_up_high(x)
    return x
    
    
class Up_layers_seconde(tf.keras.layers.Layer): ###old is num_filters//8
  def __init__(self, num_filters=256):
    super(Up_layers_seconde, self).__init__()
    self.hyper_low_up_high = tfc.SignalConv2D(num_filters//2, (3,3), name="hyper_low_up_high", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="hyper_low_up_high"))
    
    self.decode_low_up_high = tfc.SignalConv2D(num_filters//2, (3,3), name="decode_low_up_high", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="decode_low_up_high"))
    

 
  def call(self, x):
    x = self.hyper_low_up_high(x) ### put the hyper_low to become the same dimension  of the hyper_high
    x = self.decode_low_up_high(x)
    return x





class SideInfoReconModelLoad(tf.keras.layers.Layer): ###old is num_filters//8
  def __init__(self, num_filters=256):
    super(SideInfoReconModelLoad, self).__init__()
    self.hyper_low_up_high = tfc.SignalConv2D(num_filters//8, (3,3), name="hyper_low_up_high", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="hyper_low_up_high"))
    
    self.decode_low_up_high = tfc.SignalConv2D(num_filters//8, (3,3), name="decode_low_up_high", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="decode_low_up_high"))
    
    self.layer_1 = tfc.SignalConv2D(num_filters//64, (3,3), name="layer_1", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="layer_1"))

    self.layer_2 = tfc.SignalConv2D(num_filters//64, (3,3), name="layer_2", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="layer_2"))
    
    self.layer_3 = tfc.SignalConv2D(num_filters//64, (3,3), name="layer_3", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="layer_3"))

    self.layer_4 = tfc.SignalConv2D(num_filters//64, (3,3), name="layer_4", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="layer_4"))
    
    
    self.layer_5 = tfc.SignalConv2D(num_filters//64, (3,3), name="layer_5", corr=False, strides_up=1, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=tfc.GDN(name="layer_5"))
    
    self.layer_6 = tfc.SignalConv2D(3, (3,3), name="layer_3", corr=False, strides_up=1, padding="same_zeros", use_bias=True, kernel_parameterizer=parameterizers.RDFTParameterizer(), activation=None)
 
  def call(self, hyper_low, hyper_high, decode_low, decode_high):
    hyper_low = self.hyper_low_up_high(hyper_low) ### put the hyper_low to become the same dimension  of the hyper_high
    hyper_low_high = tf.concat([hyper_low, hyper_high], -1) #put the hyper_low and high together
    hyper_low_high = self.layer_1(hyper_low_high) #up 2
    hyper_low_high = self.layer_2(hyper_low_high) #up 2
    hyper_low_high = self.layer_3(hyper_low_high) #up 2
    hyper_low_high = self.layer_4(hyper_low_high) #up 2   #put the hyper_low_high have the same dimension  of the decode 
    
    decode_low = self.decode_low_up_high(decode_low) ### put the decode_low to become the same dimension  of the decode_high
    decode_low_high = tf.concat([decode_low, decode_high], -1) #put the decode_low and decode_high together
    decode_image = tf.concat([decode_low_high, hyper_low_high], -1) #put the decode_low_high and hyper_low_high together
    decode_image = self.layer_5(decode_image)
    decode_image = self.layer_6(decode_image)
    return decode_image

def train(args):
  """Trains the model."""

  tf.logging.set_verbosity(tf.logging.INFO)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  # Create input data pipeline.
  with tf.device("/cpu:0"):
    dirs = os.listdir(args.train_root_dir)
    train_files=[]
    for dir in dirs:
      path = os.path.join(args.train_root_dir, dir)
      if os.path.isdir(path):
        train_files += glob.glob(path+'/*.png')
      if os.path.isfile(path):
        train_files.append(path)
    #train_files = glob.glob(args.train_glob)
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))
    print("Number of images for training:", len(train_files))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(read_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(64)

  num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  # Instantiate model.
  encoder = Encoder(args.num_filters)
  decoder = Decoder(args.num_filters)
  hyper_encoder = HyperEncoder(args.num_filters)
  hyper_decoder = HyperDecoder(args.num_filters)
  hyper_encoder_second = HyperEncoder_second(args.num_filters)
  hyper_decoder_second = HyperDecoder_second(args.num_filters) 
  entropy_bottleneck_h = tfc.EntropyBottleneck()
  entropy_bottleneck_l = tfc.EntropyBottleneck()
  side_information_module=SideInfoReconModelLoad(args.num_filters)
  # Build autoencoder and hyperprior.
  y_h, y_l = encoder(x)
  z_h, z_l = hyper_encoder((y_h, y_l))
  
  #add second hyperior
  z_h_2, z_l_2 = hyper_encoder_second((z_h, z_l)) 
  z_h_tilde_2, z_h_likelihoods_2 = entropy_bottleneck_h(z_h_2, training=True)
  z_l_tilde_2, z_l_likelihoods_2 = entropy_bottleneck_l(z_l_2, training=True)
  hyper_param_h_2, hyper_param_l_2 = hyper_decoder_second([z_h_tilde_2, z_l_tilde_2])
  
  
  #z_h_tilde, z_h_likelihoods = entropy_bottleneck_h(z_h, training=True)
  #z_l_tilde, z_l_likelihoods = entropy_bottleneck_l(z_l, training=True)
  
  #hyper_param_h, hyper_param_l = hyper_decoder([z_h_tilde, z_l_tilde])

  ######## auto-regressive + hyper model (masked convolution)
  # if args.autoregressive:
    # entropy_param_h = EntropyParam(args.num_filters//2)
    # entropy_param_l = EntropyParam(args.num_filters//2)
    # yn_h = tf.math.add_n([y_h, tf.random.uniform(tf.shape(y_h), -0.5, 0.5)])
    # yn_l = tf.math.add_n([y_l, tf.random.uniform(tf.shape(y_l), -0.5, 0.5)])
    # ctx_param_h = maskedConv2d(yn_h, args.num_filters//2, scope="maskedConv2d_h")
    # ctx_param_l = maskedConv2d(yn_l, args.num_filters//2, scope="maskedConv2d_l")
    # mean_h, sigma_h = entropy_param_h(tf.concat([ctx_param_h, hyper_param_h], 3))
    # mean_l, sigma_l = entropy_param_l(tf.concat([ctx_param_l, hyper_param_l], 3))
  # else:
    # mean_h, sigma_h, mean_l, sigma_l = hyper_param_h[:,:,:,:args.num_filters//2], hyper_param_h[:,:,:,args.num_filters//2:], hyper_param_l[:,:,:,:args.num_filters//2], hyper_param_l[:,:,:,args.num_filters//2:]
	
  mean_l_2, sigma_l_2 = hyper_param_l_2[:,:,:,:args.num_filters//2], hyper_param_l_2[:,:,:,args.num_filters//2:]
  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))

  #conditional_bottleneck_h_2 = tfc.GaussianConditional(mean=mean_h_2, scale=sigma_h_2, scale_table=scale_table)
  conditional_bottleneck_l_2 = tfc.GaussianConditional_second(mean=mean_l_2, scale=sigma_l_2, scale_table=scale_table)
  #z_h_tilde, z_h_likelihoods = conditional_bottleneck_h_2(z_h, training=True)
  z_l_tilde, z_l_likelihoods = conditional_bottleneck_l_2(z_l, training=True)
  
  conv_z_l = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_z_l", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  z_l_tilde_up = conv_z_l(z_l_tilde)
  conv_z_l_if = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_z_l_if", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  z_l_tilde_up_if = conv_z_l_if(z_l_tilde)

  entropy_param_h_2 = EntropyParam_second(args.num_filters//2)
  mean_h_2, sigma_h_2 = entropy_param_h_2(tf.concat([z_l_tilde_up, hyper_param_h_2], 3))
  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck_h_2 = tfc.GaussianConditional_second(mean=mean_h_2, scale=sigma_h_2, scale_table=scale_table)
  z_h_tilde, z_h_likelihoods = conditional_bottleneck_h_2(z_h, training=True)
  #x_tilde = decoder([y_h_tilde,y_l_tilde])
  # decoder_h, decoder_l = decoder([y_h_tilde,y_l_tilde])
  # x_tilde = side_information_module(hyper_param_l, hyper_param_h, decoder_l, decoder_h)
  
  
  
  #First hyper layer
  
  hyper_param_h, hyper_param_l = hyper_decoder([z_h_tilde, z_l_tilde])
  
  mean_l, sigma_l = hyper_param_l[:,:,:,:args.num_filters//2], hyper_param_l[:,:,:,args.num_filters//2:]

  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))

  conv_mean_l = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_mean_l", corr=False, strides_up=1, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  conv_sigma_l = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_sigma_l", corr=False, strides_up=1, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  
  
  meal_up=Up_layers(args.num_filters)
  mean_l_2 = meal_up(mean_l_2)
  sigma_l_up=Up_layers(args.num_filters)
  sigma_l_2 = sigma_l_up(sigma_l_2)
  mean_l = conv_sigma_l(tf.concat([mean_l,mean_l_2],3))
  sigma_l = conv_sigma_l(tf.concat([sigma_l,sigma_l_2],3))
  conditional_bottleneck_l = tfc.GaussianConditional(mean=mean_l, scale=sigma_l, scale_table=scale_table)
  y_l_tilde, y_l_likelihoods = conditional_bottleneck_l(y_l, training=True)
  
  
  conv_y_l = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_y_l", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  y_l_tilde_up = conv_y_l(y_l_tilde)
  conv_y_l_if = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_y_l_if", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  y_l_tilde_up_if = conv_y_l_if(y_l_tilde)
  
  
  Up_hyper_param_h = Up_layers_seconde(args.num_filters)
  hyper_param_h_2= Up_hyper_param_h(hyper_param_h_2)
  hyper_param_h = tf.concat([hyper_param_h_2, hyper_param_h], 3)
  entropy_param_h = EntropyParam(args.num_filters//2)
  mean_h, sigma_h = entropy_param_h(tf.concat([y_l_tilde_up, hyper_param_h], 3))
  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck_h = tfc.GaussianConditional(mean=mean_h, scale=sigma_h, scale_table=scale_table)
  y_h_tilde, y_h_likelihoods = conditional_bottleneck_h(y_h, training=True)
  
  
  #x_tilde = decoder([y_h_tilde,y_l_tilde])
  decoder_h, decoder_l = decoder([y_h_tilde,y_l_tilde])
  Up_hyper_param_l = Up_layers_seconde(args.num_filters)
  hyper_param_l_2= Up_hyper_param_l(hyper_param_l_2)
  hyper_param_l = tf.concat([hyper_param_l,hyper_param_l_2],-1)
  hyper_param_h = tf.concat([hyper_param_h,hyper_param_h_2],-1)
  x_tilde = side_information_module(hyper_param_l, hyper_param_h, decoder_l, decoder_h)

  # Total number of bits divided by number of pixels. #add second hyper layer
  train_bpp_h = (tf.reduce_sum(tf.log(y_h_likelihoods)) + tf.reduce_sum(tf.log(z_h_likelihoods_2)) + tf.reduce_sum(tf.log(z_h_likelihoods))) / (-np.log(2) * num_pixels)
  train_bpp_l = (tf.reduce_sum(tf.log(y_l_likelihoods)) + tf.reduce_sum(tf.log(z_l_likelihoods)) + tf.reduce_sum(tf.log(z_l_likelihoods_2))) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2
  #information fidelity loss term
  train_if_y = tf.reduce_mean(tf.squared_difference(y_l_tilde_up_if, y_h_tilde))
  train_if_z = tf.reduce_mean(tf.squared_difference(z_l_tilde_up_if, z_h_tilde))
  step = tf.train.get_or_create_global_step()
  boundaries = [200000]
  values = [args.if_weight, 0.0]
  if_weight = tf.train.piecewise_constant(step, boundaries, values)

  # The rate-distortion cost.
  train_loss = args.lmbda * train_mse + train_bpp_h + train_bpp_l + if_weight*(train_if_y+train_if_z)

  vars_save = [var for var in tf.global_variables()]
  model_vars = [var for var in tf.global_variables()]
  vars_restore = [var for var in tf.global_variables()]
  saver = tf.train.Saver(vars_save)
  # Minimize loss and auxiliary loss, and execute update op.
  
  if args.lr_scheduling:
    boundaries = [args.last_step//2, args.last_step*6//10, args.last_step*7//10, args.last_step*8//10, args.last_step*9//10, args.last_step]
    values = [args.lr / (2 ** i) for i in range(len(boundaries) + 1)]
    lr = tf.train.piecewise_constant(step, boundaries, values)
  else:
    lr = args.lr
  main_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
  grads = main_optimizer.compute_gradients(train_loss, var_list=model_vars)
  for i, (g, v) in enumerate(grads):
    if g is not None:
      grads[i] = (tf.clip_by_value(g, -0.01 / lr, 0.01 / lr), v)
  main_step = main_optimizer.apply_gradients(grads)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=lr*10)
  entropy_bottleneck_losses = entropy_bottleneck_h.losses[0] + entropy_bottleneck_l.losses[0]
  grads = aux_optimizer.compute_gradients(entropy_bottleneck_losses, var_list=model_vars)
  for i, (g, v) in enumerate(grads):
    if g is not None:
      grads[i] = (tf.clip_by_value(g, -0.01 / (lr * 10), 0.01 / (lr * 10)), v)
  aux_step = aux_optimizer.apply_gradients(grads)

  train_op = tf.group(main_step, aux_step, entropy_bottleneck_h.updates[0], entropy_bottleneck_l.updates[0])
  with tf.control_dependencies([train_op]):
    train_op = tf.assign_add(step, 1)  

  ori = x * 255
  rec = tf.round(tf.clip_by_value(x_tilde, 0, 1) * 255)

  #mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  train_psnr = tf.squeeze(tf.reduce_mean(tf.image.psnr(ori, rec, 255)))
  train_msssim = tf.squeeze(tf.reduce_mean(tf.image.ssim_multiscale(ori, rec, 255)))

  sess.run(tf.global_variables_initializer())
  summary_list = []
  summary_list.append(tf.summary.scalar("loss", train_loss))
  summary_list.append(tf.summary.scalar("bpp_h", train_bpp_h))
  summary_list.append(tf.summary.scalar("bpp_l", train_bpp_l))
  summary_list.append(tf.summary.scalar("mse", train_mse))
  summary_list.append(tf.summary.scalar("info_fidelity", train_if_y+train_if_z))
  summary_list.append(tf.summary.scalar("psnr", train_psnr))
  summary_list.append(tf.summary.scalar("msssim", train_msssim))
  summary_list.append(tf.summary.scalar("lr", lr))

  # tf.summary.image("original", quantize_image(x))
  # tf.summary.image("reconstruction", quantize_image(x_tilde))

  merged = tf.summary.merge(summary_list)
  twriter = tf.summary.FileWriter(args.checkpoint_dir, graph=sess.graph)
  if args.restore_path is not None:
    saver_0 = tf.train.Saver(vars_restore)
    print(f'Loading learned model from checkpoint {args.restore_path}')
    aver_0.restore(sess, args.restore_path)

  step_run = 0
  while step_run < args.last_step:
    _, summary, step_run = sess.run([train_op, merged, step])
    if (step_run % 10000 == 0) or (step_run == args.last_step - 1):
      saver.save(sess, args.checkpoint_dir + f'/model.ckpt-{step_run + 1}')
    if step_run % 100 == 0:
      twriter.add_summary(summary, step_run)

################## Code for testing the model (includes two functions: encode and decode)

def encode(args, images_info, save_output=False):
  """encode an image."""

  # Load input image and add batch dimension.
  images_padded_numpy, size = images_info
  real_height_start, real_height_end, real_width_start, real_width_end, height, width = size
  with tf.name_scope('Data'):
    images_padded = tf.placeholder(tf.float32, shape=(1, height, width, 3), name='images_ori')

  # x = read_png(args.input_file)
  # x = tf.expand_dims(x, 0)
  # x.set_shape([1, None, None, 3])
  x = images_padded
  x_shape = tf.shape(x)


  # Instantiate model.
  encoder = Encoder(args.num_filters)
  decoder = Decoder(args.num_filters)
  hyper_encoder = HyperEncoder(args.num_filters)
  hyper_decoder = HyperDecoder(args.num_filters)
  hyper_encoder_second = HyperEncoder_second(args.num_filters)
  hyper_decoder_second = HyperDecoder_second(args.num_filters) 
  entropy_bottleneck_h = tfc.EntropyBottleneck()
  entropy_bottleneck_l = tfc.EntropyBottleneck()
  side_information_module=SideInfoReconModelLoad(args.num_filters)

  # Build autoencoder and hyperprior.
  y_h_input, y_l_input = encoder(x)
  z_h_input, z_l_input = hyper_encoder((y_h_input, y_l_input))
  
  #add second hyperior
  z_h_2_input, z_l_2_input = hyper_encoder_second((z_h_input, z_l_input))

  with tf.name_scope('Data'):
    y_h_input_0 = tf.placeholder(tf.float32, shape=y_h_input.shape.as_list(), name='y_h_input_0')
    y_l_input_0 = tf.placeholder(tf.float32, shape=y_l_input.shape.as_list(), name='y_l_input_0')
    z_h_input_0 = tf.placeholder(tf.float32, shape=z_h_input.shape.as_list(), name='z_h_input_0')
    z_l_input_0= tf.placeholder(tf.float32, shape=z_l_input.shape.as_list(), name='z_l_input_0')
    z_h_2_input_0 = tf.placeholder(tf.float32, shape=z_h_2_input.shape.as_list(), name='z_h_2_input_0')
    z_l_2_input_0 = tf.placeholder(tf.float32, shape=z_l_2_input.shape.as_list(), name='z_l_2_input_0')
  y_h = y_h_input_0
  y_l = y_l_input_0
  z_h = z_h_input_0
  z_l = z_l_input_0
  z_h_2 = z_h_2_input_0
  z_l_2 = z_l_2_input_0
  y_h_shape = tf.shape(y_h)
  y_l_shape = tf.shape(y_l)

  
  z_h_tilde_2, z_h_likelihoods_2 = entropy_bottleneck_h(z_h_2, training=False)
  z_l_tilde_2, z_l_likelihoods_2 = entropy_bottleneck_l(z_l_2, training=False)
  hyper_param_h_2, hyper_param_l_2 = hyper_decoder_second([z_h_tilde_2, z_l_tilde_2])
  
  
  #z_h_tilde, z_h_likelihoods = entropy_bottleneck_h(z_h, training=True)
  #z_l_tilde, z_l_likelihoods = entropy_bottleneck_l(z_l, training=True)
  
  #hyper_param_h, hyper_param_l = hyper_decoder([z_h_tilde, z_l_tilde])

  ######## auto-regressive + hyper model (masked convolution)
  # if args.autoregressive:
    # entropy_param_h = EntropyParam(args.num_filters//2)
    # entropy_param_l = EntropyParam(args.num_filters//2)
    # yn_h = tf.math.add_n([y_h, tf.random.uniform(tf.shape(y_h), -0.5, 0.5)])
    # yn_l = tf.math.add_n([y_l, tf.random.uniform(tf.shape(y_l), -0.5, 0.5)])
    # ctx_param_h = maskedConv2d(yn_h, args.num_filters//2, scope="maskedConv2d_h")
    # ctx_param_l = maskedConv2d(yn_l, args.num_filters//2, scope="maskedConv2d_l")
    # mean_h, sigma_h = entropy_param_h(tf.concat([ctx_param_h, hyper_param_h], 3))
    # mean_l, sigma_l = entropy_param_l(tf.concat([ctx_param_l, hyper_param_l], 3))
  # else:
    # mean_h, sigma_h, mean_l, sigma_l = hyper_param_h[:,:,:,:args.num_filters//2], hyper_param_h[:,:,:,args.num_filters//2:], hyper_param_l[:,:,:,:args.num_filters//2], hyper_param_l[:,:,:,args.num_filters//2:]
	
  mean_l_2, sigma_l_2 = hyper_param_l_2[:,:,:,:args.num_filters//2], hyper_param_l_2[:,:,:,args.num_filters//2:]
  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))

  #conditional_bottleneck_h_2 = tfc.GaussianConditional(mean=mean_h_2, scale=sigma_h_2, scale_table=scale_table)
  conditional_bottleneck_l_2 = tfc.GaussianConditional_second(mean=mean_l_2, scale=sigma_l_2, scale_table=scale_table)
  #z_h_tilde, z_h_likelihoods = conditional_bottleneck_h_2(z_h, training=True)
  z_l_tilde, z_l_likelihoods = conditional_bottleneck_l_2(z_l, training=False)
  
  conv_z_l = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_z_l", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  z_l_tilde_up = conv_z_l(z_l_tilde)
  conv_z_l_if = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_z_l_if", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  z_l_tilde_up_if = conv_z_l_if(z_l_tilde)
  
  entropy_param_h_2 = EntropyParam_second(args.num_filters//2)
  mean_h_2, sigma_h_2 = entropy_param_h_2(tf.concat([z_l_tilde_up, hyper_param_h_2], 3))
  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck_h_2 = tfc.GaussianConditional_second(mean=mean_h_2, scale=sigma_h_2, scale_table=scale_table)
  z_h_tilde, z_h_likelihoods = conditional_bottleneck_h_2(z_h, training=False)
  
  #First hyper layer
  
  hyper_param_h, hyper_param_l = hyper_decoder([z_h_tilde, z_l_tilde])
  
  mean_l, sigma_l = hyper_param_l[:,:,:,:args.num_filters//2], hyper_param_l[:,:,:,args.num_filters//2:]

  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))

  conv_mean_l = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_mean_l", corr=False, strides_up=1, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  conv_sigma_l = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_sigma_l", corr=False, strides_up=1, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  
  
  meal_up=Up_layers(args.num_filters)
  mean_l_2 = meal_up(mean_l_2)
  sigma_l_up=Up_layers(args.num_filters)
  sigma_l_2 = sigma_l_up(sigma_l_2)
  mean_l = conv_sigma_l(tf.concat([mean_l,mean_l_2],3))
  sigma_l = conv_sigma_l(tf.concat([sigma_l,sigma_l_2],3))
  conditional_bottleneck_l = tfc.GaussianConditional(mean=mean_l, scale=sigma_l, scale_table=scale_table)
  y_l_tilde, y_l_likelihoods = conditional_bottleneck_l(y_l, training=False)
  
  
  conv_y_l = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_y_l", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  y_l_tilde_up = conv_y_l(y_l_tilde)
  conv_y_l_if = tfc.SignalConv2D(args.num_filters//2, (5,5), name="conv_y_l_if", corr=False, strides_up=2, padding="same_zeros", use_bias=True, kernel_parameterizer=None, activation=None)
  y_l_tilde_up_if = conv_y_l_if(y_l_tilde)
  
  
  Up_hyper_param_h = Up_layers_seconde(args.num_filters)
  hyper_param_h_2= Up_hyper_param_h(hyper_param_h_2)
  hyper_param_h = tf.concat([hyper_param_h_2, hyper_param_h], 3)
  entropy_param_h = EntropyParam(args.num_filters//2)
  mean_h, sigma_h = entropy_param_h(tf.concat([y_l_tilde_up, hyper_param_h], 3))
  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck_h = tfc.GaussianConditional(mean=mean_h, scale=sigma_h, scale_table=scale_table)
  y_h_tilde, y_h_likelihoods = conditional_bottleneck_h(y_h, training=False)
  
  
  x_tilde = decoder([y_h_tilde,y_l_tilde])
  # decoder_h, decoder_l = decoder([y_h_tilde,y_l_tilde])
  # Up_hyper_param_l = Up_layers_seconde(args.num_filters)
  # hyper_param_l_2= Up_hyper_param_l(hyper_param_l_2)
  # hyper_param_l = tf.concat([hyper_param_l,hyper_param_l_2],-1)
  # hyper_param_h = tf.concat([hyper_param_h,hyper_param_h_2],-1)
  # x_tilde = side_information_module(hyper_param_l, hyper_param_h, decoder_l, decoder_h)
  
  
  
  

  side_string_h = entropy_bottleneck_h.compress(z_h_2)
  side_string_l = entropy_bottleneck_l.compress(z_l_2)
  string_h_2 = conditional_bottleneck_h_2.compress(z_h)
  string_l_2 = conditional_bottleneck_l_2.compress(z_l)
  string_h = conditional_bottleneck_h.compress(y_h)
  string_l = conditional_bottleneck_l.compress(y_l)

  # Decode the quantized image back (if requested).
  #y_h_tilde, y_h_likelihoods = conditional_bottleneck_h(y_h, training=False)
  #y_l_tilde, y_l_likelihoods = conditional_bottleneck_l(y_l, training=False)

  #x_tilde = decoder([y_h_tilde,y_l_tilde])
  
  x_tilde = x_tilde[:, :x_shape[1], :x_shape[2], :]
  
  #x_ori = x[:, top_end:top_end + height_ori, left_end:left_end + width_ori, :]
  x_ori = x[:, real_height_start:real_height_end, real_width_start:real_width_end, :]

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x_ori)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp_h = (tf.reduce_sum(tf.log(y_h_likelihoods)) + tf.reduce_sum(tf.log(z_h_likelihoods)) + tf.reduce_sum(tf.log(z_h_likelihoods_2))) / (-np.log(2) * num_pixels)
  eval_bpp_l = (tf.reduce_sum(tf.log(y_l_likelihoods)) + tf.reduce_sum(tf.log(z_l_likelihoods)) + tf.reduce_sum(tf.log(z_l_likelihoods_2))) / (-np.log(2) * num_pixels)


  #
  #x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
  #op = write_png("testencoded.png", x_hat[0, :x_shape[0], :x_shape[1], :])
  #

  # Bring both images back to 0..255 range.
  x *= 255
  x_tilde = tf.clip_by_value(x_tilde, 0, 1)
  x_tilde = tf.round(x_tilde * 255)

  # x_ori = x[:, top_end:top_end + height_ori, left_end:left_end + width_ori, :]
  # x_hat_ori = x_tilde[:, top_end:top_end + height_ori, left_end:left_end + width_ori, :]
  x_ori = x[:, real_height_start:real_height_end, real_width_start:real_width_end, :]
  x_hat_ori = x_tilde[:, real_height_start:real_height_end, real_width_start:real_width_end, :]
  mse = tf.reduce_mean(tf.squared_difference(x_ori, x_hat_ori))
  psnr = tf.squeeze(tf.image.psnr(x_ori, x_hat_ori, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_ori, x_hat_ori, 255))

  vars_restore = [var for var in tf.global_variables()]


  with tf.Session() as sess:
    # Load the latest model checkpoint, get the encoded string and the tensor
    # shapes.
    assert (args.checkpoint_dir is not None)
    """
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    """
    
    if args.checkpoint_dir is not None:
      saver_0 = tf.train.Saver(vars_restore)
      print(f'Loading learned model from checkpoint {args.checkpoint_dir}')
      saver_0.restore(sess, args.checkpoint_dir)
    
    
    y_h_test, y_l_test, z_h_test, z_l_test, z_h_2_test, z_l_2_test = sess.run(
      [y_h_input, y_l_input, z_h_input, z_l_input, z_h_2_input, z_l_2_input],
      feed_dict={images_padded: images_padded_numpy / 255.0})
    tensors = [string_h, side_string_h, string_h_2, tf.shape(x)[1:-1], tf.shape(x_ori)[1:-1], tf.shape(y_h)[1:-1], tf.shape(z_h)[1:-1], tf.shape(z_h_2)[1:-1], string_l, side_string_l, string_l_2, tf.shape(y_l)[1:-1], tf.shape(z_l)[1:-1], tf.shape(z_l_2)[1:-1]]
    arrays = sess.run(tensors,feed_dict={images_padded:images_padded_numpy/255.0, y_h_input_0:y_h_test,y_l_input_0:y_l_test,z_h_input_0:z_h_test,z_l_input_0:z_l_test,z_h_2_input_0:z_h_2_test,z_l_2_input_0:z_l_2_test})

    # Write a binary file with the shape information and the encoded string.
    packed = tfc.PackedTensors()
    packed.pack(tensors, arrays)
    if save_output:
      with open(args.output_file, "wb") as f:
        f.write(packed.string)

    # If requested, decode the quantized image back and measure performance.
    eval_bpp_h_test, eval_bpp_l_test, mse_test, psnr_test, msssim_test, num_pixels_test, x_hat_ori_test = sess.run([eval_bpp_h, eval_bpp_l, mse, psnr, msssim, num_pixels, x_hat_ori],feed_dict={images_padded:images_padded_numpy/255.0, y_h_input_0:y_h_test,y_l_input_0:y_l_test,z_h_input_0:z_h_test,z_l_input_0:z_l_test,z_h_2_input_0:z_h_2_test,z_l_2_input_0:z_l_2_test})

    x_hat_ori_test = x_hat_ori_test[0, :, :, :].astype(np.uint8)
    # Write reconstructed image out as a PNG file.
    if save_output:
      x_hat_ori_test_pil = Image.fromarray(x_hat_ori_test)
      x_hat_ori_test_pil.save(args.output_file.replace('.bitstream', '_rec.png'))

    psnr_yuv444, psnr_yuv444_list, msssim_y = evaluate(x_hat_ori_test,images_padded_numpy[0, real_height_start:real_height_end, real_width_start:real_width_end, :].astype(np.uint8))
    ### save the decoded image used for measuring the perofrmance
    #im = Image.fromarray(x_hat[0].astype(np.uint8))
    #im.save("testencoded.png")
    ###

    # The actual bits per pixel including overhead.
    bpp = len(packed.string) * 8 / num_pixels_test

    print("Mean squared error: {:0.4f}".format(mse_test))
    print("RGB PSNR (dB): {:0.2f}".format(psnr_test))
    print("YUV444 PSNR (dB): {:0.2f}".format(psnr_yuv444))
    print("RGB Multiscale SSIM: {:0.4f}".format(msssim_test))
    print("RGB Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_test)))
    print("Y Multiscale SSIM: {:0.4f}".format(msssim_y))
    print("Y Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_y)))
    print("Information content in bpp_h: {:0.4f}".format(eval_bpp_h_test))
    print("Information content in bpp_l: {:0.4f}".format(eval_bpp_l_test))
    print("Information content in bpp: {:0.4f}".format(eval_bpp_h_test+eval_bpp_l_test))
    print("Actual bits per pixel: {:0.4f}\n".format(bpp))

  tf.reset_default_graph()
  return [psnr_test, psnr_yuv444_list, msssim_test, msssim_y, bpp]


def perimage_performance(metrics_list):
  psnr_rgb_list = []
  psnr_y_list = []
  psnr_u_list = []
  psnr_v_list = []
  msssim_rgb_list = []
  msssim_y_list = []
  bpp_list = []
  for metrics_item in metrics_list:
    psnr_rgb_list.append(metrics_item[0])
    psnr_y_list.append(metrics_item[1][0])
    psnr_u_list.append(metrics_item[1][1])
    psnr_v_list.append(metrics_item[1][2])
    msssim_rgb_list.append(metrics_item[2])
    msssim_y_list.append(metrics_item[3])
    bpp_list.append(metrics_item[4])
  bpp_avg = np.mean(bpp_list)
  #RGB_MSE_avg = np.mean([255. ** 2 / pow(10, PSNR / 10) for PSNR in psnr_rgb_list])
  #RGB_PSNR_avg = 10 * np.log10(255. ** 2 / RGB_MSE_avg)
  RGB_PSNR_avg = np.mean(psnr_rgb_list)
  Y_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_y_list])
  Y_PSNR_avg = 10 * np.log10(255 ** 2 / Y_MSE_avg)
  U_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_u_list])
  U_PSNR_avg = 10 * np.log10(255 ** 2 / U_MSE_avg)
  V_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_v_list])
  V_PSNR_avg = 10 * np.log10(255 ** 2 / V_MSE_avg)
  yuv_psnr_avg = 6.0/8.0*Y_PSNR_avg + 1.0/8.0*U_PSNR_avg + 1.0/8.0*V_PSNR_avg
  msssim_rgb_avg = np.mean(msssim_rgb_list)
  msssim_y_avg = np.mean(msssim_y_list)

  print("overall performance for one image:")
  print("RGB PSNR (dB): {:0.2f}".format(RGB_PSNR_avg))
  print("YUV444 PSNR (dB): {:0.2f}".format(yuv_psnr_avg))
  print("RGB Multiscale SSIM: {:0.4f}".format(msssim_rgb_avg))
  print("RGB Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_rgb_avg)))
  print("Y Multiscale SSIM: {:0.4f}".format(msssim_y_avg))
  print("Y Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_y_avg)))
  print("Actual bits per pixel: {:0.4f}\n".format(bpp_avg))

  return [RGB_PSNR_avg, [Y_PSNR_avg,U_PSNR_avg, V_PSNR_avg], msssim_rgb_avg, msssim_y_avg, bpp_avg]

def overall_performance(metrics_list):
  psnr_rgb_list = []
  psnr_y_list = []
  psnr_u_list = []
  psnr_v_list = []
  msssim_rgb_list = []
  msssim_y_list = []
  bpp_list = []
  for metrics_item in metrics_list:
    psnr_rgb_list.append(metrics_item[0])
    psnr_y_list.append(metrics_item[1][0])
    psnr_u_list.append(metrics_item[1][1])
    psnr_v_list.append(metrics_item[1][2])
    msssim_rgb_list.append(metrics_item[2])
    msssim_y_list.append(metrics_item[3])
    bpp_list.append(metrics_item[4])
  bpp_avg = np.mean(bpp_list)
  #RGB_MSE_avg = np.mean([255. ** 2 / pow(10, PSNR / 10) for PSNR in psnr_rgb_list])
  #RGB_PSNR_avg = 10 * np.log10(255. ** 2 / RGB_MSE_avg)
  RGB_PSNR_avg = np.mean(psnr_rgb_list)
  Y_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_y_list])
  Y_PSNR_avg = 10 * np.log10(255 ** 2 / Y_MSE_avg)
  U_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_u_list])
  U_PSNR_avg = 10 * np.log10(255 ** 2 / U_MSE_avg)
  V_MSE_avg = np.mean([255 ** 2 / pow(10, PSNR / 10) for PSNR in psnr_v_list])
  V_PSNR_avg = 10 * np.log10(255 ** 2 / V_MSE_avg)
  yuv_psnr_avg = 6.0/8.0*Y_PSNR_avg + 1.0/8.0*U_PSNR_avg + 1.0/8.0*V_PSNR_avg
  msssim_rgb_avg = np.mean(msssim_rgb_list)
  msssim_y_avg = np.mean(msssim_y_list)

  print("overall performance")
  print("RGB PSNR (dB): {:0.2f}".format(RGB_PSNR_avg))
  print("YUV444 PSNR (dB): {:0.2f}".format(yuv_psnr_avg))
  print("RGB Multiscale SSIM: {:0.4f}".format(msssim_rgb_avg))
  print("RGB Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_rgb_avg)))
  print("Y Multiscale SSIM: {:0.4f}".format(msssim_y_avg))
  print("Y Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim_y_avg)))
  print("Actual bits per pixel: {:0.4f}\n".format(bpp_avg))


def decode(args):
  """ Decode an image."""

  # Read the shape information and encoded string from the binary file.
  string_h = tf.placeholder(tf.string, [1])
  side_string_h = tf.placeholder(tf.string, [1])
  string_l = tf.placeholder(tf.string, [1])
  side_string_l = tf.placeholder(tf.string, [1])
  
  side_string_l_2 = tf.placeholder(tf.string, [1])
  side_string_h_2 = tf.placeholder(tf.string, [1])
  x_shape = tf.placeholder(tf.int32, [2])
  y_h_shape = tf.placeholder(tf.int32, [2])
  z_h_shape = tf.placeholder(tf.int32, [2])
  y_l_shape = tf.placeholder(tf.int32, [2])
  z_l_shape = tf.placeholder(tf.int32, [2])
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
#  tensors = [string_h, side_string_h, x_shape, y_h_shape, z_h_shape, string_l, side_string_l, y_l_shape, z_l_shape]
    
  tensors = [string_h, side_string_h, string_h_2, x_shape, y_h_shape, z_h_shape, z_h_2_shape,string_l, side_string_l, string_l_2, y_l_shape, z_l_shape, z_l_2_shape]
  arrays = packed.unpack(tensors)

  # Instantiate model.
  decoder = Decoder(args.num_filters)
  hyper_decoder = HyperDecoder(args.num_filters)
  hyper_decoder_second = HyperDecoder_second(args.num_filters) 
  entropy_bottleneck_h = tfc.EntropyBottleneck(dtype=tf.float32)
  entropy_bottleneck_l = tfc.EntropyBottleneck(dtype=tf.float32)

  # Decode the image back.
  z_h_shape = tf.concat([z_h_shape, [args.num_filters//2]], axis=0)
  z_l_shape = tf.concat([z_l_shape, [args.num_filters//2]], axis=0)
  
  z_h_2_shape = tf.concat([z_h_2_shape, [args.num_filters//2]], axis=0)
  z_l_2_shape = tf.concat([z_l_2_shape, [args.num_filters//2]], axis=0)

  z_h_hat_2 = entropy_bottleneck_h.decompress(side_string_h, z_h_2_shape, channels=args.num_filters//2)
  z_l_hat_2 = entropy_bottleneck_l.decompress(side_string_l, z_l_2_shape, channels=args.num_filters//2)

  hyper_param_h_2, hyper_param_l_2 = hyper_decoder_second((z_h_hat_2, z_l_hat_2))
  mean_h_2, sigma_h_2, mean_l_2, sigma_l_2 = hyper_param_h_2[:,:,:,:args.num_filters//2], hyper_param_h_2[:,:,:,args.num_filters//2:], hyper_param_l_2[:,:,:,:args.num_filters//2], hyper_param_l_2[:,:,:,args.num_filters//2:]  
  mean_h_2 = mean_h_2[:, :z_h_shape[0], :z_h_shape[1], :] 
  mean_l_2 = mean_l_2[:, :z_l_shape[0], :z_l_shape[1], :] 

  sigma_h_2 = sigma_h_2[:, :z_h_shape[0], :z_h_shape[1], :]
  sigma_l_2 = sigma_l_2[:, :z_l_shape[0], :z_l_shape[1], :]

  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck_h_2 = tfc.GaussianConditional(mean=mean_h_2, scale=sigma_h_2, scale_table=scale_table, dtype=tf.float32)
  conditional_bottleneck_l_2 = tfc.GaussianConditional(mean=mean_l_2, scale=sigma_l_2, scale_table=scale_table, dtype=tf.float32)
  y_h_hat_2 = conditional_bottleneck_h_2.decompress(string_h)
  y_l_hat_2 = conditional_bottleneck_l_2.decompress(string_l)
  
  
  hyper_param_h, hyper_param_l = hyper_decoder((y_h_hat_2, y_l_hat_2))
  mean_h, sigma_h, mean_l, sigma_l= hyper_param_h[:,:,:,:args.num_filters//2], hyper_param_h[:,:,:,args.num_filters//2:], hyper_param_l_2[:,:,:,:args.num_filters//2], hyper_param_l[:,:,:,args.num_filters//2:]  
  mean_h = mean_h[:, :y_h_shape[0], :y_h_shape[1], :] 
  mean_l = mean_l[:, :y_l_shape[0], :y_l_shape[1], :] 

  sigma_h = sigma_h[:, :y_h_shape[0], :y_h_shape[1], :]
  sigma_l = sigma_l[:, :y_l_shape[0], :y_l_shape[1], :]

  scale_table = np.exp(np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck_h = tfc.GaussianConditional(mean=mean_h, scale=sigma_h, scale_table=scale_table, dtype=tf.float32)
  conditional_bottleneck_l = tfc.GaussianConditional(mean=mean_l_2, scale=sigma_l_2, scale_table=scale_table, dtype=tf.float32)
  y_h_hat = conditional_bottleneck_h.decompress(string_h)
  y_l_hat = conditional_bottleneck_l.decompress(string_l)
  
  
  x_hat = decoder((y_h_hat, y_l_hat))

  # Remove batch dimension, and crop away any extraneous padding on the bottom or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = write_png(args.output_file, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op, feed_dict=dict(zip(tensors, arrays)))


def get_splitted_images_list(input_file):
  I = Image.open(input_file)
  I_array = np.array(I)
  height_ori, width_ori, _ = np.shape(I_array)
  height = height_ori
  width = width_ori
  images_list = []
  if width_ori > 5000:
    height = (height_ori // 128) * 128 if height_ori % 128 == 0 else (height_ori // 128 + 1) * 128
    top_end = (height - height_ori) // 2
    real_height_start = top_end
    real_height_end = top_end + height_ori
    width = width_ori//2
    width = (width // 128) * 128 if width % 128 == 0 else (width // 128 + 1) * 128
    I_array_padded = np.zeros((1, height, width, 3), np.uint8)
    I_array_padded[0,real_height_start:real_height_end,:,:] = I_array[:,:width,:]
    real_width_start = 0
    real_width_end = width_ori//2
    images_list.append([I_array_padded,(real_height_start, real_height_end, real_width_start, real_width_end, height, width)])
    I_array_padded1 = np.zeros((1, height, width, 3), np.uint8)
    I_array_padded1[0, real_height_start:real_height_end, :, :] = I_array[:, width_ori-width:width_ori, :]
    real_width_start = width-width_ori//2
    real_width_end = width
    images_list.append([I_array_padded1,(real_height_start, real_height_end, real_width_start, real_width_end, height, width)])
    print('height_pad:', height, 'width_pad:', width)
    return images_list
  if height_ori > 5000:
    width = (width_ori // 128) * 128 if width_ori % 128 == 0 else (width_ori // 128 + 1) * 128
    left_end = (width - width_ori) // 2
    real_width_start = left_end
    real_width_end = left_end + width_ori
    height = height_ori // 2
    height = (height // 128) * 128 if height % 128 == 0 else (height // 128 + 1) * 128
    I_array_padded = np.zeros((1, height, width, 3), np.uint8)
    I_array_padded[0, :, real_width_start:real_width_end, :] = I_array[:height, :, :]
    real_height_start = 0
    real_height_end = height_ori // 2
    images_list.append([I_array_padded,(real_height_start, real_height_end, real_width_start, real_width_end, height, width)])
    I_array_padded1 = np.zeros((1, height, width, 3), np.uint8)
    I_array_padded1[0, :, real_width_start:real_width_end, :] = I_array[height_ori-height:height_ori, :, :]
    real_height_start = height - height_ori // 2
    real_height_end = height
    images_list.append([I_array_padded1,(real_height_start, real_height_end, real_width_start, real_width_end, height, width)])
    print('height_pad:', height, 'width_pad:', width)
    return images_list

  height = (height_ori // 128) * 128 if height_ori % 128 == 0 else (height_ori // 128 + 1) * 128
  width = (width_ori // 128) * 128 if width_ori % 128 == 0 else (width_ori // 128 + 1) * 128
  top_end = (height - height_ori) // 2
  left_end = (width - width_ori) // 2
  real_height_start = top_end
  real_height_end = top_end + height_ori
  real_width_start = left_end
  real_width_end = left_end + width_ori
  I_array_padded = np.zeros((1,height,width,3), np.uint8)
  I_array_padded[0,top_end:top_end+height_ori, left_end:left_end+width_ori,:]=I_array
  print('height_pad:', height, 'width_pad:', width)
  return [[I_array_padded,(real_height_start, real_height_end, real_width_start, real_width_end, height, width)]]

def get_image_size(input_file):
  I = Image.open(input_file)
  I_array = np.array(I)
  height_ori, width_ori, _ = np.shape(I_array)
  height = (height_ori // 128) * 128 if height_ori % 128 == 0 else (height_ori // 128 + 1) * 128
  width = (width_ori // 128) * 128 if width_ori % 128 == 0 else (width_ori // 128 + 1) * 128
  top_end = (height - height_ori) // 2
  left_end = (width - width_ori) // 2
  I_array_padded = np.zeros((1,height,width,3), np.uint8)
  I_array_padded[0,top_end:top_end+height_ori, left_end:left_end+width_ori,:]=I_array
  print('height_pad:', height, 'width_pad:', width)
  return I_array_padded,(height_ori, width_ori, height, width)


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument("--autoregressive", "-AR", action="store_true", help="Include autoregressive model for training")
  parser.add_argument("--num_filters", type=int, default=192, help="Number of filters per layer.")
  parser.add_argument("--restore_path", default=None, help="Directory where to load model checkpoints.")
  parser.add_argument("--checkpoint_dir", default="train", help="Directory where to save/load model checkpoints.")
  parser.add_argument("--if_weight", type=int, default=1.0, help="weights")
  subparsers = parser.add_subparsers(title="commands", dest="command",
      help="commands: 'train' loads training data and trains (or continues "
           "to train) a new model. 'encode' reads an image file (lossless "
           "PNG format) and writes a encoded binary file. 'decode' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Trains (or continues to train) a new model.")
  train_cmd.add_argument("--train_root_dir", default="images", help="The root directory of training data, which contains a list of RGB images in PNG format.")
  train_cmd.add_argument("--batchsize", type=int, default=8, help="Batch size for training.")
  train_cmd.add_argument("--patchsize", type=int, default=256, help="Size of image patches for training.")
  train_cmd.add_argument("--lambda", type=float, default=0.01, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument("--last_step", type=int, default=1000000, help="Train up to this number of steps.")
  train_cmd.add_argument("--lr", type=float, default = 1e-4, help="Learning rate [1e-4].")
  train_cmd.add_argument("--lr_scheduling", "-lr_sch", action="store_true", help="Enable learning rate scheduling, [enabled] as default")
  train_cmd.add_argument("--preprocess_threads", type=int, default=16, help="Number of CPU threads to use for parallel decoding of training images.")

  # 'encode' subcommand.
  encode_cmd = subparsers.add_parser("encode", formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Reads a PNG file, encode it, and writes a 'bitstream' file.")
  # 'decode' subcommand.
  decode_cmd = subparsers.add_parser("decode",formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="Reads a 'bitstream' file, reconstructs the image, and writes back a PNG file.")

  # Arguments for both 'encode' and 'decode'.
  for cmd, ext in ((encode_cmd, ".bitstream"), (decode_cmd, ".png")):
    cmd.add_argument("input_file", help="Input filename.")
    cmd.add_argument("output_file", nargs="?", help="Output filename (optional). If not provided, appends '{}' to the input filename.".format(ext))

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args

def main(args):
  # Invoke subcommand.
  os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
  if args.command == "train":
    train(args)
  elif args.command == "encode": # encoding
    if not args.output_file:
      args.output_file = args.input_file + ".bitstream"
    if os.path.isdir(args.input_file):
      dirs = os.listdir(args.input_file)
      test_files = []
      for dir in dirs:
        path = os.path.join(args.input_file, dir)
        if os.path.isdir(path):
          test_files += glob.glob(path + '/*.png')[:6]
        if os.path.isfile(path):
          test_files.append(path)
      if not test_files:
        raise RuntimeError(
          "No testing images found with glob '{}'.".format(args.input_file))
      print("Number of images for testing:", len(test_files))
      metrics_list=[]
      for file_idx in range(len(test_files)):
        file = test_files[file_idx]
        print(str(file_idx)+" testing image:", file)
        args.input_file = file
        #file_name = file.split('/')[-1]
        #args.output_file = args.output_file + file_name.replace('.png', '.bitstream')
        splitted_images_list = get_splitted_images_list(args.input_file)
        metrics_temp_list = []
        for subimage_idx in range(len(splitted_images_list)):
          sub_image = splitted_images_list[subimage_idx]
          print(str(subimage_idx)+" subimage")
          metrics = encode(args, sub_image)
          metrics_temp_list.append(metrics)
        metrics_perimage = perimage_performance(metrics_temp_list)
        metrics_list.append(metrics_perimage)
      overall_performance(metrics_list)
    else:
      image_padded, size = get_image_size(args.input_file)
      metrics = encode(args, image_padded, size, True)
  elif args.command == "decode": # decoding
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decode(args)

if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
