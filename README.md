# Introduction
This project mainly dedicated to show how [Google Compression](https://github.com/tensorflow/compression), in which they introduced context-based entropy model for
image compression using deep learning, realize their work! And their websie lies in [tensorflow.github.io/compression](tensorflow.github.io/compression), and IN THE VERY BEGINING OF THE END, more description of there work shows in my [blog](https://mluthervandross.github.io/mlv/)💪🏿💪🏿💪🏿!!!

# Installation
***Note: Precompiled packages are currently only provided for Linux (Python 2.7, 3.3-3.6) and Darwin/Mac OS (Python 2.7, 3.7). To use these packages on Windows, consider using a TensorFlow Docker image and installing tensorflow-compression using pip inside the [Docker container](https://www.tensorflow.org/install/docker).***

## MacOs/Linux
***Using Anaconda***
It seems that Anaconda ships its own binary version of TensorFlow which is incompatible with our pip package. It also installs Python 3.7 by default, which we currently don't support on Linux. To solve this, make sure to use Python 3.6 on Linux, and always install TensorFlow via pip rather than conda. For example, this creates an Anaconda environment with Python 3.6 and CUDA libraries, and then installs TensorFlow and tensorflow-compression with GPU support:
```
conda create --name ENV_NAME python=3.6 cudatoolkit=10.0 cudnn
conda activate ENV_NAME
pip install tensorflow-gpu==1.15
```

***Furthermore***
For correctly using my precompiled binary pip package, you must first download the wheel file and then install it with
```
pip install tensorflow_compression-*.whl
```

# Usage
We recommend importing the library from your Python code as follows:
```
import tensorflow as tf
import tensorflow_compression as tfc
```


# Citation
[Joint Autoregressive and Hierarchical Priors for Learned Image Compression](https://arxiv.org/abs/1809.02736)
[Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436)
[End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704)
[Density Modeling of Images using a Generalized Normalization Transformation](https://arxiv.org/abs/1511.06281)
