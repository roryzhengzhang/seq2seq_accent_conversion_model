from genericpath import isfile
import os
import glob
from random import shuffle

pairs = []
refer_name = 'bdl'
mode = 'bnf'
root = os.path.dirname(os.path.abspath(__file__))

for f in glob.glob(os.path.join(root, f"data_wav_16k/{refer_name}/*.wav")):
    filename = f.split('/')[-1]
    audioname = filename[len(refer_name)+1:-4]

    for speaker in ['clb', 'EBVS', 'ERMS', 'MBMPS', 'NJS', 'rms', 'slt', 'bdl']:

        src_wav_emb = f'{mode}_emb/{refer_name}/{refer_name}_{audioname}.npy'
        tar_mel_file = f'mel_emb/{speaker}/{speaker}_{audioname}.npy'
        speaker_emb = f'speaker_emb/{speaker}/{speaker}_{audioname}.npy'
        accent_emb = f'accent_emb/{speaker}/{speaker}_{audioname}.npy'

        if os.path.isfile(tar_mel_file):
            pairs.append([src_wav_emb, tar_mel_file, speaker_emb, accent_emb])
    
shuffle(pairs)
    
with open(f'{mode}_train_pairs.txt', 'w') as train_file, open(f'{mode}_val_pairs.txt', 'w') as val_file:
    for i, pair in enumerate(pairs):
        if i < len(pairs)-50:
            train_file.write(f"{pair[0]},{pair[1]},{pair[2]},{pair[3]}\n")
        else:
            val_file.write(f"{pair[0]},{pair[1]},{pair[2]},{pair[3]}\n")
