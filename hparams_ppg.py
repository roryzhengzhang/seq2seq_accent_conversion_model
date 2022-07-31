import tensorflow as tf
from text import symbols

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

"""
    tf.contrib.training.HParams get removed from tf2, replace it with AttrDict
"""
def create_hparams(hparams_string=None, verbose=False):
    hparams = AttrDict({

        ################################
        # PPGMelLoader Hyperparameters #
        ################################

        "is_full_ppg": True,  # Whether to use the full PPG or not.
        "is_append_f0": False,  # Currently only effective at sentence level
        "ppg_subsampling_factor": 1,  # Sub-sample the ppg & acoustic sequence.
        # Cases
        # |'load_feats_from_disk'|'is_cache_feats'|Note
        # |True                  |True            |Error
        # |True                  |False           |Please set cache path
        # |False                 |True            |Overwrite the cache path
        # |False                 |False           |Ignores the cache path
        "load_feats_from_disk": True,  # Remember to set the path.
        # Mutually exclusive with 'load_feats_from_disk', will overwrite
        # 'feats_cache_path' if set.
        "is_cache_feats": False,
        "feats_cache_path": '',
        "n_acoustic_feat_dims": 80,

        ################################
        # Experiment Parameters        #
        ################################
        "epochs": 2000,
        "iters_per_checkpoint": 500,
        "seed": 1234,
        "dynamic_loss_scaling": True,
        "fp16_run": False,
        "distributed_run": False,
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "ignore_layers": ['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        "load_mel_from_disk": False,
        "training_files": 'ppg_train_pairs.txt',
        "validation_files": 'ppg_val_pairs.txt',
        "text_cleaners": ['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        "max_wav_value": 32768.0,
        "sampling_rate": 16000,
        "filter_length": 1024,
        "hop_length": 160,
        "win_length": 1024,
        "n_mel_channels": 80,
        "n_bnf_channel": 1,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,

        ################################
        # Model Parameters             #
        ################################
        "n_symbols": 2000,
        "symbols_embedding_dim": 512,

        # Encoder parameters
        "encoder_kernel_size": 5,
        "encoder_n_convolutions": 3,
        "encoder_embedding_dim": 1024,
        "decoder_input_dim": 1024,

        # Decoder parameters
        "n_frames_per_step": 1,  # currently only 1 is supported
        "decoder_rnn_dim": 1024,
        "prenet_dim": 256,
        "max_decoder_steps": 1000,
        "gate_threshold": 0.5,
        "p_attention_dropout": 0.1,
        "p_decoder_dropout": 0.1,

        # Attention parameters
        "attention_rnn_dim": 1024,
        "attention_dim": 128,

        # Location Layer parameters
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,

        # Mel-post processing network parameters
        "postnet_embedding_dim": 512,
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 5,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate": False,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "grad_clip_thresh": 1.0,
        "lr_decay": 0.995,
        "step_size": 1,
        "batch_size": 48,
        "mask_padding": True,  # set model's padded outputs to padded values

        "audio_dir": "audio_data_16k",
        "speaker_embedding_dir": "speaker",
        "accent_embedding_dir": "accent",
        "use_accent_emb": False,
        "use_speaker_emb": False,
    })

    return hparams
