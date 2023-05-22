import csv
import numpy as np
import pandas as pd
import librosa as lb
import glob
from librosa import feature
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
from mutagen.wave import WAVE
from maad.features import temporal_events

fn_list_ii = [
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff
]

fn_list_i = [
    feature.spectral_flatness,
    feature.zero_crossing_rate
]


def get_feature_vector(y, sr):
    f_v = np.array([])
    for funct in fn_list_ii:
        matrix = funct(y=y, sr=sr)
        res_vec = np.mean(matrix, axis=1)
        f_v = np.concatenate((f_v, res_vec))

    for funct in fn_list_i:
        matrix = funct(y=y)
        res_vec = np.mean(matrix, axis=1)
        f_v = np.concatenate((f_v, res_vec))

    return f_v


participants = pd.read_excel("PsychiatricDiscourse_participant_data.xlsx")
schizophrenia_only = participants.loc[
    (participants['depression.symptoms'] == 0.) &
    (participants['thought.disorder.symptoms'] != 0.)
]

wav_files_dir = '**/*.wav'
wav_files = glob.glob(wav_files_dir)  # how to drop 'wav_files/'?

feature_vector = []
for file in wav_files:
    if file[8:14] in list(schizophrenia_only.ID):
        y, sr = lb.load(file, sr=44100)
        sound = AudioSegment.from_mp3(file)
        audio_chunks = split_on_silence(sound, min_silence_len=250, silence_thresh=-40)
        silence = detect_silence(sound, min_silence_len=250, silence_thresh=-40)
        audio = WAVE(file)
        rel_num_pause = (len(audio_chunks) - 1) / audio.info.length
        mean_length_pause = np.mean([b - a for a, b in silence]) / 1000
        mean_spoken_ratio, mean_utterance_duration, _, _ = temporal_events(y, fs=sr, dB_threshold=-29, rejectDuration=0.5)
        f0 = lb.yin(y=y, sr=sr, fmin=80, fmax=175)
        f0_std = np.std(f0)
        fv = get_feature_vector(y, sr)
        fv = list(fv)
        fv.insert(0, file[8:14])
        fv.insert(1, file.split("-")[2])
        fv.insert(2, schizophrenia_only.loc[(schizophrenia_only['ID'] == file[8:14]), 'thought.disorder.symptoms'].item())
        fv += [rel_num_pause, mean_length_pause, audio.info.length, mean_spoken_ratio, mean_utterance_duration,
               f0_std]
        feature_vector.append(fv)


norm_output = 'voice_features_red.csv'


header = ["id", "stimulus", "symptoms_score", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
          "spectral_flatness", "zero_crossing_rate", "rel_num_pause", "mean_length_pause", "len",
          "mean_spoken_ratio", "mean_utterance_duration", "f0_std"]


with open(norm_output, "+w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(header)
    csv_writer.writerows(feature_vector)

df = pd.read_csv("voice_features_red.csv")
print(df)
