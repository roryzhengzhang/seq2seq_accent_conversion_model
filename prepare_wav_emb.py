import os
import numpy as np
import glob
import soundfile as sf
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForPreTraining

wav2vec_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
wav2vec_model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")

out_dir = 'wav_emb'

for wav in sorted(glob.glob('data_wav_16k/slt/*.wav')):
    speaker = wav.split('/')[-2]
    filename = Path(wav).stem
    if not os.path.isfile(f'{out_dir}/{speaker}/{filename}.npy'):
        audio_input, _ = sf.read(wav)
        input_values = wav2vec_feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt").input_values
        hidden_vec = wav2vec_model(input_values, output_hidden_states=True, return_dict=True).hidden_states[-1]
        hidden_vec = hidden_vec.squeeze(0)
        if not os.path.isdir(f'wav_emb/{speaker}'):
            os.makedirs(f'wav_emb/{speaker}')
        with open(f'{out_dir}/{speaker}/{filename}.npy', 'wb') as f: 
            np.save(f, hidden_vec.detach().numpy())

