# Generalized Octave Convolution-based Learned Image Compression with Multi-Layer Hyper-Priors and Cross-Resolution Parameter Estimation

This repository contains the code for reproducing the results with trained models, in the following paper:

Our code is based on the paper named Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules. arXiv, CVPR2020. Zhengxue Cheng, Heming Sun, Masaru Takeuchi, Jiro Katto

Our paper is Learned Image Compression with Discretized Gaussian-Laplacian-Logistic Mixture Model and Concatenated Residual Modules. arXiv. Haisheng Fu, Feng Liang, Jianping Lin, Bing Li, Mohammad Akbari, Jie Liang, Guohe Zhang, Dong Liu, Chengjie Tu, Jingning Han

Paper Summary
Recently, image compression approaches based on deep learning have gradually outperformed existing image compression standards including BPG and VVC intra coding. In particular, the application of the context-adaptive entropy model significantly improves the rate-distortion (R-D) performance, in which hyperpriors and autoregressive models are jointly utilized to effectively capture the spatial redundancy of the latent representations. However, the latent representations still contain some spatial correlations. In addition, these methods based on the context-adaptive entropy model cannot be accelerated in the decoding process by parallel computing devices, e.g. FPGA or GPU. To alleviate these limitations, we propose a learned multi-resolution image compression framework, which exploits the recently developed octave convolutions to factorize the latent representations into the high-resolution (HR) and low-resolution (LR) parts, similar to wavelet transform, which further improves the R-D performance. To speed up the decoding, our scheme does not use context-adaptive entropy model. Instead, we exploit an additional hyper layer including hyper encoder and hyper decoder to further remove the spatial redundancy of the latent representation. Moreover, the cross-resolution parameter estimation (CRPE) is introduced into the proposed framework to enhance the flow of information and further improve the R-D performance. An additional information-fidelity loss is proposed to the total loss function to adjust the contribution of the LR part to the final bit stream. Experiments show that the proposed method outperforms the BPG (4:4:4) and some state-of-the-art learned schemes in both PSNR and MS-SSIM metrics on Kodak and Tecnick datasets. In particular, our scheme can achieve better PSNR than VVC-intra (4:2:0) on Kodak dataset at high bit rates (> 0.35 bpp) and outperforms the VVC-intra (4:2:0) across a wide range of bit rates on Tecnick dataset. Moreover, our scheme is much faster in decoding speed compared to approaches using context-adaptive entropy models.

Environment
Python==3.6.4

Tensorflow==1.14.0

RangeCoder

    pip3 install range-coder
Tensorflow-Compression ==1.2
    pip3 install tensorflow-compression or 
    pip3 install tensorflow_compression-1.2-cp36-cp36m-manylinux1_x86_64.whl
Test Usage
Download the pre-trained models (The Extraction code is i6p3. These models are optimized by PSNR using lambda = 0.0016(number filters=128)) and lambda = 0.03(number filters=256)).

Run the following py files can encode or decode the input file.

   python Encoder_Decoder_cvpr_blocks_leaky_GLLMM_directly_bits_github.py
   note that:
   endcoder_main(); // the Encoder code
   decoder_main();  // the Decoder  code
   path ='xxx';     // the test image 
   save_image_name_path=''; // save the bit stream files.
   num_filters = 128 or 256;  // 128 for low bit rates and 256 for high bit rates.
   
Reconstructed Samples
Comparisons of reconstructed samples are given in the following.



Evaluation Results


Notes
If you think it is useful for your reseach, please cite our paper.

@misc{fu2021learned,
      title={Learned Image Compression with Discretized Gaussian-Laplacian-Logistic Mixture Model and Concatenated Residual Modules}, 
      author={Haisheng Fu and Feng Liang and Jianping Lin and Bing Li and Mohammad Akbari and Jie Liang and Guohe Zhang and Dong Liu and Chengjie Tu and Jingning Han},
      year={2021},
      eprint={2107.06463},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
