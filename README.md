This repository contains a PyTorch implementation of the original feed-forward Neural Style Transfer paper  ([🔗Johnson et al.](https://arxiv.org/pdf/1603.08155.pdf))

**What's the different between Real-Time Style Transfer and Image Optimization Style Transfer?**

The image optimization style transfer is an algorithm that doesn't change the network parameters weight, it only changes the pixels values of the random initialized image. See my previews [repository](https://github.com/hovav698/Style-Transfer-Image-Optimization-) for more details. Therefore this algorithm can only make style trasfer for a specific content image, but it doesn't learn general patterns for converting any random content image for the corresponding style transfer image. For any new content image we need to run the algorithm and optimize the image pixels values again. 
This method isn't suited for making style transfer for videos, because it will require to reoptimized the image weight for each frame of the video, and it can take very long time.

The Real Time Style Transfer algorithm solves this problem. It trained on a large iage dataset, and learn how to add style for any content image. After the model has been trained, getting styled imaged is much faster comparing to the previews method, and it allows to easily get style transfered video.

**How the the algorithm works?**

The algorithms consist of two models:

**Transformer-Net network** (Doesn't have any connection to the NLP transformer model): 

The network transforms input images into output images. The exact model architecture can be found [here](https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf). The images that we will input into this network will be from the coco14 image dataset. This network is resposible to generate the styled images for any image that fed into it. 

<img width="516" alt="RTST" src="https://user-images.githubusercontent.com/71300410/121808335-9a256200-cc60-11eb-8f7e-09b214af1a87.PNG">


**VGG16 based CNN network**

The vgg based The network outputs activations from different layers of the vgg network, each activation represent features of different scale.

**Loss calculation and backpropogation**

In similar to the image optimization style transfer algorithm, this algorithm will also consist of both content and style loss function and will use the gram matrix for style calculation. The process will be as follow:

The images from the dataset will be input into the Transformer-Net, it will generate a new image according to it's current model weight. The generated image, the original image and the style image will then be input into the vgg based model to extract the relevant feature. The loss function will receive the outputs from the VGG model, and a backpropogation process will start to minimize the loss by adjusting the Transformer-Net weight. Only the Transformer-Net model weight will be updated, the vgg based model weight will be fixed.

**Last step - creating video**

In the end of the process the model will learn the style pattern of the style image, and can output style transfer image for any random image.
I want to perform style transfer for video, so I will first extract all the frames of a sample video, and insert all the frames into the Transformer-Net. It will then add style to the frame according to the style image. In the end of the process i will reunite all the frame and make from it a style transfer video.

**Results**








