import os
import glob
import random
from venv import create
from hparams import create_hparams
from utils.common import hparams
from utils.common.data_utils import PPGMelLoader


with open('filelist.txt', 'w') as f:
    audiofiles = []
    for audio in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_wav_16k/*/*.wav')):
        audiofiles.append(audio)

    random.seed(1234)
    random.shuffle(audiofiles)

    for audio in audiofiles:
        f.write(audio+'\n')


hparams = create_hparams()
loader = PPGMelLoader( hparams, data_utterance_paths='filelist.txt')