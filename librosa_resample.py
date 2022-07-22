import librosa
import glob
import os
from pathlib import Path
import soundfile as sf

speaker = "rms"
for f in glob.glob(f"/Users/test/Downloads/cmu_us_{speaker}_arctic/wav/*.wav"):
    y, sr = librosa.load(f)
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
    out_path = f"data_wav_16k/{speaker}/"
    Path(out_path).mkdir(parents=True, exist_ok=True)
    sf.write(os.path.join(out_path, f"{speaker}_{Path(f).stem}.wav"), y_16k, 16000)
