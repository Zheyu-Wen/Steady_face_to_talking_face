import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import glob
import cv2
from frame2vedio import f2v


def train(net, dataloader, nepoch):
    loss_fun = nn.SmoothL1Loss()
    loss_hist = []
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=5e-4)
    for iters in range(nepoch):
        loss_in_iter = 0
        num_of_bacth = 0
        for wav, img, fixed_img in dataloader:
            num_of_bacth += 1
            batch_size = wav.shape[0]
            wav = wav.view(batch_size, 1, 12, 34)
            pred_face = net(wav, fixed_img)
            optimizer.zero_grad()
            loss = loss_fun(pred_face, img)
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.detach().cpu().numpy())
            loss_in_iter += loss.detach().cpu().numpy()
            pred_face = pred_face.permute(0, 2, 3, 1).detach().cpu().numpy()
            SaveImage(pred_face, pred_face.shape[0])
        print('In training epoch: {}, loss is: {}'.format(iters+1, loss_in_iter/num_of_bacth))
        if (iters + 1) % 5 == 0:
            if not os.path.exists("network_model"):
                os.makedirs("network_model")
            torch.save(net.state_dict(), f"network_model/model_{iters + 1}.pth")
    return loss_hist


def SaveImage(batch_fake_image, batch_size):
    for i in range(batch_size):
        new_im = Image.fromarray(np.uint8(np.clip(batch_fake_image[i, :, :, :], 0, 1) * 255))
        new_im.save("test_{}.png".format(i))
    # tail = '.png'
    # png_file = sorted(glob.glob(f'test*{tail}'))
    # frames = np.zeros([batch_size, 224, 224, 3])
    # for i in range(batch_size):
    #     frames[i, :, :, :] = cv2.imread(png_file[i])
    f2v(batch_fake_image)
