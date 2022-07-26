import random
import numpy as np
import torch
import torch.utils.data
import soundfile as sf
import layers
import os
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence

from transformers import AutoFeatureExtractor, AutoModelForPreTraining

class AudioDataset(torch.utils.data.Dataset):
    """
        1) loads audio
        2) computes mel-spectrograms from audio files
        3) generate lingusitic vector from wav2vec
        4) generate speaker and accent dvector from audio
    """
    def __init__(self, filepath, hparams):
        super(AudioDataset, self).__init__()
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        self.max_wav_value = hparams.max_wav_value
        self.wav2vec_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.wav2vec_model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.filepath = filepath
        self.speaker_embedding_dir = hparams.speaker_embedding_dir
        self.accent_embedding_dir = hparams.accent_embedding_dir
        # self.src_audio_list = []
        self.tar_audio_list = []
        self.src_wav_embs = []
        self.speaker_info = []
        self.accent_info = []

        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                splits = line.split(',')
                src_wav_emb, tar_wav, speaker, accent = splits[0], splits[1], splits[2], splits[3]
                # filename = os.path.join(hparams.audio_dir, filename)
                # self.src_audio_list.append(splits[0])
                self.src_wav_embs.append(src_wav_emb)
                self.tar_audio_list.append(tar_wav)
                self.speaker_info.append(speaker)
                self.accent_info.append(accent)

    def __getitem__(self, index):
        return self.get_vec_mel_speaker_accent_pair(index)

    def __len__(self):
        return len(self.src_wav_embs)

    def get_vec_mel_speaker_accent_pair(self, index):
        tar_wav = self.tar_audio_list[index]
        wav_vec = self.load_wav_embedding(self.src_wav_embs[index])
        speaker_emb = self.speaker_info[index]
        accent_emb = self.accent_info[index]
        mel = self.get_mel(tar_wav)
        speaker_vector = self.load_speaker_embedding(speaker_emb)
        accent_vector = self.load_accent_embedding(accent_emb)

        return (wav_vec, mel, speaker_vector, accent_vector)

    def load_speaker_embedding(self, speaker_emb):
        # return torch.from_numpy(np.load(os.path.join(self.speaker_embedding_dir, f"{speaker}.npy")))
        return torch.from_numpy(np.load(speaker_emb))

    def load_accent_embedding(self, accent_emb):
        # return torch.from_numpy(np.load(os.path.join(self.accent_embedding_dir, f"{accent}.npy")))
        return torch.from_numpy(np.load(accent_emb))

    def load_wav_embedding(self, wav_emb):
        return torch.from_numpy(np.load(wav_emb))

    def wav2vec(self, filename):
        audio_input, _ = sf.read(filename)
        input_values = self.wav2vec_feature_extractor(audio_input, sampling_rate=self.sampling_rate, return_tensors="pt").input_values
        hidden_vec = self.wav2vec_model(input_values, output_hidden_states=True, return_dict=True).hidden_states[-1]
        hidden_vec = hidden_vec.squeeze(0).detach()
        return hidden_vec
        

    def get_mel(self, filename):
        """
            return mel in range of [-1, 1]
        """
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        
        return melspec

class AudioCollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """
            Collate's training batch from BNF and mel-spectrogram
        """
        # Right zero-pad all one-hot text sequences to max input length
        bnf_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = bnf_lengths[0]
        bnf_feature_size = batch[0][0][2].size(0)
        bnf_padded = torch.FloatTensor(len(batch), max_input_len, bnf_feature_size)
        bnf_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            bnf = batch[ids_sorted_decreasing[i]][0]
            bnf_padded[i, :bnf.size(0)] = bnf

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
        

        speaker_emb = torch.stack([x[2] for x in batch], dim=0)
        # speaker_feature_size = speaker_emb.size(1)
        # speaker_padded = torch.FloatTensor(len(batch), max_input_len, speaker_feature_size)
        # speaker_padded.zero_()
        # for i in range(len(ids_sorted_decreasing)):
        #     bnf = batch[ids_sorted_decreasing[i]][0]
        #     speaker_padded[i, :bnf.size(0)] = speaker_emb[ids_sorted_decreasing[i]]
        

        accent_emb = torch.stack([x[3] for x in batch], dim=0)
        # accent_feature_size = accent_emb.size(1)
        # accent_padded = torch.FloatTensor(len(batch), max_input_len, accent_feature_size)
        # accent_padded.zero_()
        # for i in range(len(ids_sorted_decreasing)):
        #     bnf = batch[ids_sorted_decreasing[i]][0]
        #     speaker_padded[i, :bnf.size(0)] = accent_emb[ids_sorted_decreasing[i]]

        return bnf_padded, bnf_lengths, mel_padded, gate_padded, \
            output_lengths, speaker_emb, accent_emb



class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths
