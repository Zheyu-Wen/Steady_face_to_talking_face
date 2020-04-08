import cv2
import numpy as np
import glob
import os
from PIL import Image

def f2v(frames, save_dir):
    # create output video
    height = 224
    width = 224
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(save_dir, fourcc, 5, (height, width))

    for i in range(frames.shape[0]):
        # scale the frame back to 0-255
        frame = (frames[i]).astype(np.uint8)
        # write frame to output video
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()

def SaveImage(image_file):
    batch_size = len(image_file)
    for i in range(batch_size):
        new_im = Image.fromarray(np.uint8(np.load(image_file[i])*255))
        new_im.save("{}.png".format(image_file[i][:-4]))

def change_name_style(path):
    all_file = os.listdir(path)
    for item in all_file:
        if item[-9] == 'O':
            new_name = item[:-8] + '0' + item[-8:]
            os.rename(path + '/' + item, path + '/' + new_name)

if __name__=='__main__':
    path = 'dataset/jpgnpy'
    tails = 'npy'
    test_file = sorted(glob.glob(f'{path}/_4YV5Z6jNWY*{tails}'))
    SaveImage(test_file)
    tail = '.png'
    png_file = sorted(glob.glob(f'{path}/_4YV5Z6jNWY*{tail}'))
    frames = np.zeros([len(test_file), 224, 224, 3])
    for i in range(len(test_file)):
        frames[i, :, :, :] = cv2.cvtColor(cv2.imread(png_file[i]), cv2.COLOR_BGR2RGB)
    f2v(frames, 'test.avi')
