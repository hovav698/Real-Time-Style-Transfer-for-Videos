from utils.utils import get_data, gram_matrix, get_image, transform, extract_video_frames, numeric_files_sort, \
    create_style_transfer_video
from models.vgg16 import Vgg16
from models.transformerNet import TransformerNet
import torch
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# calculation of the content loss - it minimize the difference between the content image and the generated image
def calc_content_loss(gen_feat, orig_feat):
    content_l = torch.mean((gen_feat - orig_feat) ** 2)
    return content_l


# the loss function will minimize the different between the gramm matrix of the style image
# and the content image
def calc_style_loss(gen, style):
    batch_size, channel, height, width = gen.shape

    A = gram_matrix(style[0])
    style_loss = 0
    for n in range(batch_size):
        G = gram_matrix(gen[n])
        style_loss += torch.mean((G - A) ** 2)
    return style_loss


# the total loss calculation - combination of the style loss and the content loss
def calculate_loss(gen_features, orig_feautes, style_featues, alpha, beta):
    style_loss = content_loss = 0

    for gen, cont in zip(gen_features.relu2_2, orig_feautes.relu2_2):
        content_loss += calc_content_loss(gen, cont)

    for gen, style in zip(gen_features, style_featues):
        style_loss += calc_style_loss(gen, style)

    total_loss = alpha * content_loss + beta * style_loss
    return total_loss


# the train loop
def train(alpha, beta, images_save_path):
    losses = []
    # alpha represent how much we want to preserve the original content of the image
    # beta represent how much we style we want to add to the content image

    for epoch in range(epochs):
        batch_count = 0

        for content_batch, _ in train_loader:
            batch_count += 1
            content_batch = content_batch.to(device)

            # Feed content batch through transformer net
            gen_batch = gen_model(content_batch)

            # Feed content and stylized batch through perceptual net (VGG16) and normalize by the imagenet mean
            gen_batch = vgg(gen_batch.add(imagenet_neg_mean))
            orig_batch = vgg(content_batch.add(imagenet_neg_mean))
            style_batch = vgg(style_img.add(imagenet_neg_mean))

            # calculate the total loss
            total_loss = calculate_loss(gen_batch, orig_batch, style_batch, alpha, beta)

            # do a backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

            # print the losses and save the test images to see what the model has learned
            if batch_count % 2 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_count}/{n_batchs} Loss: {total_loss.item()}')
                img = get_image(test_img_path, gen_model)
                plt.figure(figsize=(10, 5))
                plt.imshow(img)
                file_path = images_save_path + 'epoch' + str(epoch) + 'batch' + str(batch_count) + '.png'
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                return losses

    return losses


# The function transform and save all the frames of the video
def perform_style_transfer(frames_path):
    # count the number of frames in the video
    files = glob(frames_path + '/*')
    frames_count = len(files)

    for i in range(frames_count):
        content_image_path = frames_path + 'kang' + str(i) + '.jpg'
        img = get_image(content_image_path, gen_model)
        img = cv2.cvtColor(np.array(img, dtype=np.float32) * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite('Outputs/style transfer frames/' + str(i) + '.jpg', img)


if __name__ == "__main__":
    get_data()

    # We will use the test image to see the progress of the model learning process
    test_img_path = 'data/elephant.jpg'

    train_dataset = datasets.ImageFolder('data/Image_dataset', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # The model will learn the style pattern of the following image
    style_path = 'data/styles/starrynight.jpg'
    style_img = Image.open(style_path)

    style_img = transform(style_img).view(1, -1, 256, 256).to(device)

    # the original vgg model images was normalized according to the following values
    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32).reshape(1, 3, 1, 1).to(device)

    # parameters for the train loop
    epochs = 3
    lr = 0.001

    gen_model = TransformerNet().to(device)
    gen_model.train()

    vgg = Vgg16().to(device)

    # we used the adam optimizer. only the tranformer model weight will be adjusted
    optimizer = torch.optim.Adam(gen_model.parameters(), lr=lr)

    # we will train for batch size of 4
    n_files = len(glob('data/Image_dataset/train2014/*'))
    batch_size = 4
    n_batchs = n_files // batch_size

    # first we want to see what is the style pattern that the model learned by setting alpha to zero
    # the style are saved under the Pure Style Learning Process folder
    pure_style_learning_folder = 'Outputs/Pure Style Learning Process/'
    train(0, 20, pure_style_learning_folder)

    # next we choose different alphas and betas, the model will learn how to do the style transfer
    # the style transfer images are saved under the style transfer learning process

    style_transfer_learning_folder = 'Outputs/Style Transfer Learning Process/'
    losses = train(1, 30, pure_style_learning_folder)

    plt.plot(losses, label="Loss")
    plt.title("Loss")
    plt.ylim(0, 0.6e7)
    plt.legend()
    plt.savefig('Outputs/train_loss.png', dpi=300, bbox_inches='tight')

    # After the model has been trained, it's easy and fast to make a style transfer video
    # every frame in the video will be convert to an image, and each image will be transformed
    # to the style image.
    # the final stage is to unite the transformed framed to a video
    sample_video_path = 'data/sample_video.mp4'
    video_frames_path = 'Outputs/video frames/'
    style_transfered_frames = 'Outputs/style transfer frames/'

    extract_video_frames(sample_video_path, video_frames_path)

    perform_style_transfer(video_frames_path)

    # sort the frames according to it's numerical value
    files = numeric_files_sort(style_transfered_frames + '*.jpg')

    # convert the style transferred frames into mp4 video
    create_style_transfer_video(files)
