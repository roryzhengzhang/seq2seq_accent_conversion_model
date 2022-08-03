import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    # Assume the len of batches after encoder is same
    max_len = torch.max(lengths).item()

    if torch.cuda.is_available():
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    else:
        ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

def get_mask_from_lengths_window_and_time_step(lengths, attention_window_size,
                                               time_step):
    """
    One for mask and 0 for not mask
    Args:
        lengths:
        attention_window_size:
        time_step: zero-indexed

    Returns:

    """
    # Mask all initially.
    max_len = torch.max(lengths).item()
    B = len(lengths)
    if torch.cuda.is_available():
        mask = torch.cuda.ByteTensor(B, max_len)
    else:
        mask = torch.ByteTensor(B, max_len)
    mask[:] = 1

    for ii in range(B):
        # Note that the current code actually have a minor side effect,
        # where the utterances that are shorter than the longest one will
        # still have their actual last time step unmasked when the decoding
        # passes beyond that time step. I keep this bug here simply because
        # it will prevent numeric errors when computing the attention weights.
        max_idx = lengths[ii] - 1
        # >=0, <= the actual sequence end idx (length-1) (not covered here)
        start_idx = min([max([0, time_step-attention_window_size]), max_idx])
        # <=length-1
        end_idx = min([time_step+attention_window_size, max_idx])
        if start_idx > end_idx:
            continue
        mask[ii, start_idx:(end_idx+1)] = 0
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
