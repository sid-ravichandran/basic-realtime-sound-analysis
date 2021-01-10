# SCRIPT THAT RECORDS SOUND FOR A SET NUMBER OF SECONDS AND ANALYSES -
# - FREQ, BPM, NOTES AND KEY OF THE SOUND SEGMENT

import numpy as np
import pyaudio
import os
import struct
import matplotlib.pyplot as plt
import time

import Beat_detector as beat
import Freq_identifier as freq
import note_identifier as note

freq_flag, freq_plot_flag = False, False # Note identify
beat_flag, beat_plot_flag, librosa_Flag = False, False, False # find bpm
note_chuk_flag = True # divide recording into chunks of sound and analyse key of recording

pya_chunk = 1024 # PyAudio recording chunk
pya_format = pyaudio.paInt16
pya_channels = 1 # single channel recording
fs = 44100  # Sample rate
seconds = 3  # Duration of recording

p = pyaudio.PyAudio()
stream = p.open(
    format = pya_format,
    channels = pya_channels,
    rate = fs,
    input = True,
    output = True,
    frames_per_buffer = pya_chunk
)
frames = [] # A python-list of chunks(numpy.ndarray)
for i in range(0, int((fs/pya_chunk) * seconds)):
    data = stream.read(pya_chunk)
    frames.append(np.fromstring(data, dtype=np.int16))

stream.stop_stream()
stream.close()
p.terminate()

# convert data to numpy array
if pya_channels > 1:
    myrecording = np.stack((frames[::2], frames[1::2]), axis=0)
else:
    myrecording = np.hstack(frames)

if freq_flag:
    # Identify the dominant frequence content of the recording
    freqs = freq.freq_identify(myrecording, pya_channels, fs, method='periodogram', plot_flag=freq_plot_flag)
    notes = note.note_identify(freqs)
if beat_flag:
    # find the bpm of the recording
    bpm = beat.beat_detect(myrecording, pya_channels, fs, filter_flag=1, plot_flag=beat_plot_flag, librosa_flag = librosa_flag)

if note_chuk_flag:
    # Break up the sound recorded into chunks of 0.1 second and find notes. Notes are used to find the key of the recording
    # Optional - generate an idealised sine wave tone sequence based on detection of notes from the recorded sound
    chunk_length = 0.1
    n_chunks = np.floor(seconds/chunk_length)
    chunk_size = int(fs*chunk_length)
    chunk_notes = []
    chunk_freqs = []
    note_generated = []

    for i in range(int(n_chunks)):
        rec_chunk = myrecording[i*chunk_size:(i+1)*chunk_size]
        tmp_chunk = freq.freq_identify(rec_chunk, pya_channels, fs, method='periodogram', plot_flag=0)
        chunk_freqs = np.append(chunk_freqs,tmp_chunk)
        chunk_notes.append(note.note_identify(tmp_chunk))

        # note_generated = np.append(note_generated,note.note_generator(tmp_chunk, fs, chunk_length))

    # Identify key of recording
    major_key, minor_key = note.key_identifier(chunk_notes)