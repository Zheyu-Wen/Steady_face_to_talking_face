from python_speech_features import mfcc
from python_speech_features import logfbank
# import scipy.io.wavfile as wav
# import wave
import numpy as np
import os

# framerate=8000, nframes=34122,
# (rate,sig) = wav.read('english.wav')
# mfcc_feat = mfcc(sig,rate)
files= os.listdir('wavnpy')

for file in files:
    sig = np.load('wavnpy/'+file)
    mfcc_feat = mfcc(sig)
    mfcc_feat = mfcc_feat.transpose()[1:,:]
    filename = file[:-7]
    np.save('mfccwavnpy/'+filename+'mfccwav.npy', mfcc_feat)
    # print(mfcc_feat.shape)
    # print(type(mfcc_feat))
    # break
    # if mfcc_feat.shape != (12,599):
        # print(mfcc_feat.shape)

print('Done.')