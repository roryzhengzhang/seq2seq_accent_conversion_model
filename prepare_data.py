import glob
import re
from pathlib import Path

files = glob.glob('audio_data_16k/*.wav')

accent_mapping = {
    "nora": "GB"
}

with open("train_files.txt", 'w') as out: 
    for f in files:
        p = Path(f)
        path = Path(*p.parts[1:])
        speaker = re.match(r'([A-Za-z]+)', Path(f).stem)
        speaker = speaker.groups(0)[0]
        accent = accent_mapping[speaker]
        out.write(f"{path},{speaker},{accent}\n")
