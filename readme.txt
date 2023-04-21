Train or Test Aperture Patterns and Reconstruction Algorithm for a Coded Aperture Camera
====

Given light field training data, you can get optimized networks describing aperture patterns and the corresponding reconstruction algorithm.
If you use the codes for research purpose, please cite the following conference paper,

Yasutaka Inagaki, Yuto Kobayashi, Keita Takahashi, Toshiaki, Fujii, Hajime Nagahara:
"Learning to Capture Light Fields through a Coded Aperture Camera,"
European Conference on Computer Vision (ECCV), 2018


## Requirement
---------------------------------------
The included codes require Python (version 3.5.2), Chainer (version 3.2.0), CUDA (version 8.0), cuDNN (version 6.0) and Pillow.
We have tested the software on Ubuntu 16.04 LTS.


## Usage
---------------------------------------
TRAINING
Please locate the training data in "Train/Src" folder.
What is located in this folder is a temporally dataset that is different from the one we actually used in the above paper.

To train a network, execute the following command in the "Train" folder.
$ python3 train.py
Trained networks and aperture patterns will be generated in "Train/Dst" folder.

TEST
We include our trained networks and datasets as an example in "Test/Src" folder.
If you want to try other networks or datasets, please place them in the appropriate folders. 

To test the network, execute the following command in the "Test" folder.
$ python3 test.py
The reconstructed light field along with the acquired images and aperture patterns will be generated in "Test/Dst" folder.


## License
---------------------------------------
The MIT License (MIT)

Copyright (c) 2018 Yasutaka Inagaki at Fujii Lab. of Nagoya University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
