from email.mime import audio
import matplotlib
# %matplotlib inline
import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForPreTraining
# from denoiser import Denoiser

from speechbrain.pretrained import HIFIGAN

def plot_data(data, audioname, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
    fig.savefig(f'output/{audioname}-mel-spectrogram.png')

def wav2vec(filename):
    audio_input, _ = sf.read(filename)
    wav2vec_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    wav2vec_model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    input_values = wav2vec_feature_extractor(audio_input, sampling_rate=hparams.sampling_rate, return_tensors="pt").input_values
    hidden_vec = wav2vec_model(input_values, output_hidden_states=True, return_dict=True).hidden_states[-1]
    hidden_vec = hidden_vec.detach()
    return hidden_vec

hparams = create_hparams()
hparams.sampling_rate = 16000

checkpoint_path = "checkpoint/train_0803/checkpoint_3500.zip"
model = load_model(hparams)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()
else:
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
    _ = model.eval()

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

# if torch.cuda.is_available():
#     hifi_gan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").cuda()
# else:
#     hifi_gan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft", map_location=torch.device('cpu'), force_reload=True)

src_speaker = "bdl"
tar_speaker = 'bdl'
audio_name = "arctic_a0067"

speaker_emb = torch.from_numpy(np.load(f'speaker_emb/{tar_speaker}/{tar_speaker}_{audio_name}.npy')).unsqueeze(0)
accent_emb = torch.from_numpy(np.load(f'accent_emb/{src_speaker}/{src_speaker}_{audio_name}.npy')).unsqueeze(0)
wav_emb = wav2vec(f"data_wav_16k/{src_speaker}/{src_speaker}_{audio_name}.wav")
print(f"wav_emb size: {wav_emb.size()}")
mel_outputs, mel_outputs_postnet, _, alignments = model.inference((wav_emb, speaker_emb, accent_emb))
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T),
           audio_name)

np.save(f'output/{src_speaker}_{tar_speaker}_{audio_name}.npy', mel_outputs.float().data.cpu().numpy()[0])

waveforms = hifi_gan.decode_batch(mel_outputs)

# waveforms, sr = hifi_gan.generate(mel_outputs)
# print(f"sample rate: {sr}")
torchaudio.save(f'output/{src_speaker}_{tar_speaker}_{audio_name}_new.wav',waveforms.squeeze(1), 16000)