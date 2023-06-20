import os
import numpy as np
import librosa
dataset_path = r"D:\Dataset\ESC-50-master"
meta_path = os.path.join(dataset_path, "meta", "esc50.csv")
audio_path = os.path.join(dataset_path, "audio")
resample_path = r"D:\Dataset\ESC-50-master\audio_32k"
# prepare for the folder
meta = np.loadtxt(meta_path , delimiter=',', dtype='str', skiprows=1)
audio_list = os.listdir(audio_path)
# resample
for f in audio_list:
    full_f = os.path.join(audio_path, f)
    resample_f = os.path.join(resample_path, f)
    print('sox ' + full_f + ' -r 32000 ' + resample_f)
    os.system('sox ' + full_f + ' -r 32000 ' + resample_f)

output_dict = [[] for _ in range(5)]
for label in meta:
    name = label[0]
    fold = label[1]
    target = label[2]
    y, sr = librosa.load(os.path.join(resample_path, name), sr = None)
    output_dict[int(fold) - 1].append(
        {
            "name": name,
            "target": int(target),
            "waveform": y
        }
    )

np.save("esc-50-data.npy", output_dict)
