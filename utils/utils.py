import torch
import os
from tensorflow.keras.utils import get_file
import pathlib
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from glob import glob
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# download and extract the coco2014 image dataset
def get_data():
    data_dir = pathlib.Path('data/Image_dataset')
    print(data_dir.exists())
    if not data_dir.exists():
        get_file(
            'train2014.zip',
            origin="http://images.cocodataset.org/zips/train2014.zip",
            extract=True,
            cache_dir='.', cache_subdir='data/Image_dataset'
        )
        #os.remove('data/train2014.zip')



# Calculating the gram matrix for the style and the generated image.
# the gramm matrix output is the represent style of each layer of the input
def gram_matrix(input):
    channel, height, width = input.shape
    G = torch.mm(input.view(channel, height * width), input.view(channel, height * width).t())
    return G / (height * width * channel)


# The image transormation and the coco dataset preperation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

#output the generated image
def get_image(img_path, gen_model):
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)
    img = torch.clamp(img, 0, 255)
    img = gen_model(img).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = np.array(img, dtype=np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img / 255)
    return img

#extract and saves the video file frames
def extract_video_frames(source_path, target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    cap = cv2.VideoCapture(source_path)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(target_path + 'kang' + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def numeric_files_sort(path):
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    files = sorted(glob(path), key=alphanum_key)

    return files


def create_style_transfer_video(files):
    # conver the style transfered frames into mp4 video
    img_array = []
    for filename in files:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('data/Outputs/output_video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
