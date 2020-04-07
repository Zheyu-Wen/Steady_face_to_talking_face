from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
import os

class Face2Talk_Dataloader(Dataset):
    def __init__(self, path, size=224):
        super().__init__()
        wav_all = sorted(glob.glob(f"{path}/*wav.npy"))
        face_all = sorted(glob.glob(f"{path}/*jpg.npy"))
        face_wanted = [wav.replace("mfccwav", "jpg") for wav in wav_all]
        self.face = []
        for face in face_wanted:
            if face in face_all:
                self.face.append(face)

        self.wav = []
        wav_wanted = [face.replace("jpg", "mfccwav") for face in face_all]
        for wav in wav_wanted:
            if wav in wav_all:
                self.wav.append(wav)

        self.fixed_face = [face[:-8]+"0jpg.npy" for face in self.face]


        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.wav)

    def __getitem__(self, item):
        wav = np.load(self.wav[item])
        wav = wav.astype(np.float32)

        img = Image.fromarray(np.uint8(np.load(self.face[item])))
        img = self.transform(img)

        fixed_img = Image.fromarray(np.uint8(np.load(self.fixed_face[item])))
        fixed_img = self.transform(fixed_img)
        return wav, img, fixed_img