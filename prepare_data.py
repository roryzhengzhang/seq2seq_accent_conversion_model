import os
import glob
from random import shuffle

pairs = []
refer_name = 'bdl'
for f in glob.glob(f"/Users/test/Documents/GitHub/tacotron2/data_wav_16k/{refer_name}/*.wav"):
    filename = f.split('/')[-1]
    audioname = filename[len(refer_name)+1:-4]

    for speaker in ['clb', 'EBVS', 'ERMS', 'MBMPS', 'NJS', 'rms', 'slt', 'bdl']:
        src_wav_emb = f'/Users/test/Documents/GitHub/tacotron2/wav_emb/{refer_name}/{refer_name}_{audioname}.npy'
        tar_audio_file = f'/Users/test/Documents/GitHub/tacotron2/data_wav_16k/{speaker}/{speaker}_{audioname}.wav'
        speaker_emb = f'/Users/test/Documents/GitHub/tacotron2/speaker_emb/{speaker}/{speaker}_{audioname}.npy'
        accent_emb = f'/Users/test/Documents/GitHub/tacotron2/accent_emb/{speaker}/{speaker}_{audioname}.npy'

        # print(f"{tar_audio_file}, {speaker_emb}, {accent_emb}")

        if os.path.isfile(tar_audio_file):
            pairs.append([src_wav_emb, tar_audio_file, speaker_emb, accent_emb])
    
shuffle(pairs)
    
with open('train_pairs.txt', 'w') as train_file, open('val_pairs.txt', 'w') as val_file:
    for i, pair in enumerate(pairs):
        if i < len(pairs)-50:
            train_file.write(f"{pair[0]},{pair[1]},{pair[2]},{pair[3]}\n")
        else:
            val_file.write(f"{pair[0]},{pair[1]},{pair[2]},{pair[3]}\n")
