from Data_preprocess import Face2Talk_Dataloader
from Network_model import Speech2Vid
from train_process import train
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import os
import glob
from frame2vedio import f2v
from PIL import Image
import cv2

def change_name_style_jpg():
    all_file = glob.glob('test_data_set/*jpg.npy')
    for item in all_file:
        if item[-9] == 'O':
            new_name = item[:-8] + '0' + item[-8:]
            os.rename(item, new_name)

def change_name_style_wav():
    all_file = glob.glob('test_data_set/*wav.npy')
    for item in all_file:
        if item[-13] == 'O':
            new_name = item[:-12] + '0' + item[-12:]
            os.rename(item, new_name)

# change_name_style_jpg()
# change_name_style_wav()

train_mode = False
if train_mode == True:
    path = 'train_data_set'
    train_dataloader = Face2Talk_Dataloader(path)
    train_data = torch.utils.data.DataLoader(train_dataloader, batch_size=20)
    network = Speech2Vid()
    nepoch = 5
    loss_hist = train(network, train_data, nepoch)
else:
    network = Speech2Vid()
    network.load_state_dict(torch.load('network_model/model_5.pth'))
    network.eval()
    # test_file_jpg = f'{path}/_4YV5Z6jNWYNO00jpg.npy'
    path_test = 'test_data_set'
    tails = 'wav.npy'
    test_file_wav = sorted(glob.glob(f'{path_test}/*{tails}'))
    test_dataloader = Face2Talk_Dataloader(path_test)
    test_data = torch.utils.data.DataLoader(test_dataloader, batch_size=38)
    for wav, _, fixed_face in test_data:
        wav = wav.view(fixed_face.shape[0], 1, 12, 34)
        pred_image = network(wav, fixed_face)
        pred_image = pred_image.permute(0, 2, 3, 1).detach().cpu().numpy()
    for i in range(pred_image.shape[0]):
        new_im = Image.fromarray(np.uint8((pred_image[i, :, :, :])*255))
        new_im.save(test_file_wav[i][:-4]+'.png')
    tail = '.png'
    png_file = sorted(glob.glob(f'{path_test}/*{tail}'))
    frames = np.zeros([len(test_file_wav), 224, 224, 3])
    for i in range(len(test_file_wav)):
        frames[i, :, :, :] = cv2.imread(png_file[i])
    save_dir = f'{path_test}/test_video.avi'
    f2v(frames, save_dir)
