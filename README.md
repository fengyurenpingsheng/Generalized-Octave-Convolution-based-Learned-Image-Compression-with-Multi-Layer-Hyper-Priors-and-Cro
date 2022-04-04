# Generalized Octave Convolution-based Learned Image Compression with Multi-Layer Hyper-Priors and Cross-Resolution Parameter Estimation


### Paper Summary



### Environment 

* Python==3.6.4

* Tensorflow==1.14.0

* [RangeCoder](https://github.com/lucastheis/rangecoder)

```   
    pip3 install range-coder
```

* [Tensorflow-Compression](https://github.com/tensorflow/compression) ==1.2

```
    pip3 install tensorflow-compression or 
    pip3 install tensorflow_compression-1.2-cp36-cp36m-manylinux1_x86_64.whl
```
    
    
### Test Usage

* We provide an example of a test model. Download the pre-trained [models](https://pan.baidu.com/s/1VZ8EZZzX8VKJg4auKxVytQ) (The Extraction code is i6p3. The test model is optimized by PSNR using lambda = 0.02(number filters=448)).

* Run the following py files can encode or decode the input file. 

```
   python icml2020_three_layers_zhengze_noside_test.py
   note that:
   path ='xxx';     // the test image 
   save_image_name_path=''; // save the bit stream files.
   num_filters = 448;  // 256 for low bit rates and 448 for high bit rates.
   
```
### additional details

We will provided more detail information about our paper after our paper is accepted.
