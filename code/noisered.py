from scipy.io import wavfile
import noisereduce as nr
import glob

wav_files_dir = '**/*.wav'
wav_files = glob.glob(wav_files_dir)

for file in wav_files:
    print(file)
    rate, data = wavfile.read(file)
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.99, n_fft=2048, hop_length=256)
    split = file.split(sep='.')
    split[0] = split[0] + "_redacted"
    new_name = ".".join(split)
    wavfile.write(new_name, rate, reduced_noise)
