import numpy as np
import pandas as pd
import librosa as lb
import glob
from scipy.signal import hilbert

participants = pd.read_excel("PsychiatricDiscourse_participant_data.xlsx")
schizophrenia_only = participants.loc[
    (participants['depression.symptoms'] == 0.) &
    (participants['thought.disorder.symptoms'] != 0.)
    ]

wav_files_dir = '**/*.wav'
wav_files = glob.glob(wav_files_dir)  # how to drop 'wav_files/'?

feature_vector = []
counter = 0
sum = 0
for file in wav_files:
    if file[10:16] in list(schizophrenia_only.ID):
        counter += 1
        y, sr = lb.load(file, sr=44100)
        S = np.abs(lb.stft(y))
        S_db = lb.amplitude_to_db(S)
        sum = sum + np.mean(S_db)
print(sum/counter)
