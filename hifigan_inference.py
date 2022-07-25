from speechbrain.pretrained import HIFIGAN
import torchaudio
import numpy as np
import torch
import matplotlib.pylab as plt

# def plot_data(data, figsize=(16, 4)):
#     fig, axes = plt.subplots(1, len(data), figsize=figsize)
#     for i in range(len(data)):
#         axes[i].imshow(data[i], aspect='auto', origin='lower', 
#                        interpolation='none')
#     fig.savefig('fac-mel-spectrogram.png')

# hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
if torch.cuda.is_available():
    hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").cuda()
else:
    hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft", map_location='cpu', force_reload=True)

mel_outputs = torch.from_numpy(np.load(f'sample_mel/fac_syn_mel.npy')).transpose(0, 1).unsqueeze(0)

# plot_data((mel_outputs.float().data.cpu().numpy()[0]))

fig, axes = plt.subplots(1, 1)
axes.imshow(mel_outputs.float().data.cpu().numpy()[0], aspect='auto', origin='lower', 
                       interpolation='none')
fig.savefig('fac-mel-spectrogram.png')

# waveforms = hifi_gan.decode_batch(mel_outputs)
waveforms, sr = hifigan.generate(mel_outputs)
torchaudio.save(f'output/ac_syn_syn.wav',waveforms.squeeze(1), 16000)