This repository contains a PyTorch implementation of the original feed-forward Neural Style Transfer paper  ([🔗Johnson et al.](https://arxiv.org/pdf/1603.08155.pdf))

**What's the different between Real-Time Style Transfer and Image Optimization Style Transfer?**

The image optimization style transfer is an algorithm that doesn't change the network parameters weight, it only changes the pixels values of the random initialized image. See my previews [repository](https://github.com/hovav698/Style-Transfer-Image-Optimization-) for more details. As result, this algorithm can only make style trasfer for a specific content image, but it doesn't learn general patterns for converting any random image to the corresponding style transfer image. For any new content image we need to run the algorithm and optimize the image pixels values again. 
This method isn't suited for making style transfer for videos, because it will require to re-optimize the image weight for each frame of the video, and it can take very long computation time.

The Real Time Style Transfer algorithm solves this problem. It trained on a large image dataset, and learn how to add style for any image on real time. After the model has been trained, getting styled imaged is much faster comparing to the previews method, making it easy to make style transfer video.

**How the the algorithm works?**

The algorithms consist of two models:

Transformer-Net network (Doesn't have any connection to the NLP transformer model): 

The network transforms input images into output images. The exact model architecture can be found [here](https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf). We will use the coco14 image dataset as input for the network. The network is resposible to generate the styled images for any image that fed into it. 


VGG16 based CNN network:

The vgg based The network outputs activations from different layers of the vgg network, each activation represent features of different scale.

Illustration of the model architecture:
<img width="516" alt="RTST" src="https://user-images.githubusercontent.com/71300410/121808335-9a256200-cc60-11eb-8f7e-09b214af1a87.PNG">


**Loss calculation and backpropogation**

In similar to the image optimization style transfer algorithm, this algorithm will also consist of both content and style loss function and will use the gram matrix for style calculation. The process works as follows:

The images from the dataset is input into the Transformer-Net,it generates a new image according to it's current model weight. The generated image, the original image and the style image is then input into the VGG based model to extract the relevant features. The loss function receives the outputs from the VGG model, and a backpropogation process  starts to minimize the loss by adjusting the Transformer-Net weight. Only the Transformer-Net model weight are updated, the VGG based model weight stay fixed.

Example of the pure style that the model has learned (content weight set to 0):

Original Style Image    ->                    The style that the model has learned

<img src="https://user-images.githubusercontent.com/71300410/121797285-ff109600-cc27-11eb-91a9-fee190e8b734.png" width="250" height="250" />   ->  <img src="https://user-images.githubusercontent.com/71300410/121809717-9399e900-cc66-11eb-95c6-c22a03d542fb.PNG" width="250" height="250" /> 

Exmple of style transfer for a random image:

<img src="https://user-images.githubusercontent.com/71300410/121810287-e5437300-cc68-11eb-8b24-6164feb5f0d7.jpg" width="250" height="250" />   ->  <img src="https://user-images.githubusercontent.com/71300410/121810372-4ec38180-cc69-11eb-9efe-5563cf3e3dd4.png" width="250" height="250" /> 

It's clear from the style transfer image that there is a problem that causes dead pixels. I will try to understand what is the problem and fix it in the future.


**Last step - creating video**

In the end of the process the model learns the style pattern of the style image, and can output style transfer image for any random image.
I want to perform style transfer for videos, so the first step is to extract all the frames of a sample video, then input all the frames into the Transformer-Net. The syle of the style image is added to the frame accordingly. In the end of the process all the frames are reunite to create style transfer video.

**Results**

The original video:

https://user-images.githubusercontent.com/71300410/121810060-fc359580-cc67-11eb-9eec-2a4914fe69fd.mp4


The style trasfer video:


https://user-images.githubusercontent.com/71300410/121810226-a1506e00-cc68-11eb-956c-ad8309c90523.mp4


The result are not ideal, I did succeed to create style transfer video, however there is a problem of dead pixels. The style that was transfer to the video isn't obvious as in my previews style optimization project. To be continued.
