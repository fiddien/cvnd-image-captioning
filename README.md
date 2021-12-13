# Image Captioning

[![Udacity Computer Vision Nanodegree](http://tugan0329.bitbucket.io/imgs/github/cvnd.svg)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)<br/>

Given an image, the model uses CNN encoder (on top of a pre-trained [ResNet50](https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/)) and RNN decoder to produces an appropriate caption.

![](https://video.udacity-data.com/topher/2018/March/5ab588e3_image-captioning/image-captioning.png)


- The dataset used to train the network is [Microsoft Common Objects in COntext (MS COCO)](http://cocodataset.org/#home).
- A vocabulary file is created from the captions provided by CoCoDataaset using NLTK.
- A ResNet50-based layers are used to produce feature vectors for the input images.
- An RNN decoder composed of LSTM cells processed the image until generates a vector of words index.
- The model was trained using GPU in about 3 hours, with 2.2074 loss and 9.09239 perplexity.
