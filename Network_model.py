import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

class Speech2Vid(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(3, (1, 2), 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(3, 2, 1),
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.ReLU()
        )


        self.face_conv1 = nn.Conv2d(3, 96, 7, 2, 1)
        self.face_pool1 = nn.AvgPool2d(3, 2, 1)
        self.face_conv2 = nn.Conv2d(96, 256, 3, 2, 1)
        self.face_pool2 = nn.AvgPool2d(3, 2, 1)
        self.face_conv3 = nn.Conv2d(256, 512, 3, 2, 1)
        self.face_conv4 = nn.Conv2d(512, 512, 3, 2, 1)
        self.face_conv5 = nn.Conv2d(512, 512, 3, 2, 1)
        self.face_fc6 = nn.Conv2d(512, 512, 1, 1)
        self.face_fc7 = nn.Conv2d(512, 256, 1, 1)

        self.decoder_fc1 = nn.Conv2d(512, 128, 1, 1)
        self.decoder_convt2 = nn.ConvTranspose2d(128, 512, 6, 2, 1)
        self.decoder_convt3 = nn.ConvTranspose2d(512, 256, 5, 2, 1)
        self.decoder_convt3b = nn.ConvTranspose2d(256, 256, 6, 2, 1)
        self.decoder_convt4 = nn.ConvTranspose2d(512, 96, 3, 2, 1)

        self.decoder_convt5 = nn.ConvTranspose2d(192, 96, 5, 2, 1)
        self.decoder_convt6 = nn.ConvTranspose2d(96, 64, 4, 2, 1)
        self.decoder_convt7 = nn.ConvTranspose2d(64, 3, 3, 1, 0)


    def forward(self, audio, fixed_face):
        audio_encoded = self.audio_encoder(audio)
        face_encode = torch.relu(self.face_conv1(fixed_face))
        face_encode_decon5 = torch.relu(self.face_pool1(face_encode))
        face_encode_decon4 = torch.relu(self.face_conv2(face_encode_decon5))
        face_encode = torch.relu(self.face_pool2(face_encode_decon4))
        face_encode = torch.relu(self.face_conv3(face_encode))
        face_encode = torch.relu(self.face_conv4(face_encode))
        face_encode = torch.relu(self.face_conv5(face_encode))
        face_encode = torch.relu(self.face_fc6(face_encode))
        face_encoded = torch.relu(self.face_fc7(face_encode))
        face_audio_cat = torch.relu(torch.cat([audio_encoded, face_encoded], 1))
        face_decode = torch.relu(self.decoder_fc1(face_audio_cat))
        face_decode = torch.relu(self.decoder_convt2(face_decode))
        face_decode = torch.relu(self.decoder_convt3(face_decode))
        face_decode = torch.relu(self.decoder_convt3b(face_decode))
        face_encode_decode_cat4 = torch.cat([face_decode, face_encode_decon4], 1)
        face_decode = torch.relu(self.decoder_convt4(face_encode_decode_cat4))
        face_encode_decode_cat5 = torch.cat([face_decode, face_encode_decon5], 1)
        face_decode = torch.relu(self.decoder_convt5(face_encode_decode_cat5))
        face_decode = torch.relu(self.decoder_convt6(face_decode))
        face_decoded = torch.tanh(self.decoder_convt7(face_decode))

        return face_decoded

class Deblur(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblur = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 3, 3)
        )

    def forward(self, inputs):
        residual = self.deblur(inputs)
        out = torch.add(inputs, residual)
        return out