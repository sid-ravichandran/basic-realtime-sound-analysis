import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import simpleaudio as sa

import Beat_detector as beat
import Freq_identifier as freq
import note_identifier as note

freq_flag, freq_plot_flag, freq_method = True, False, 'welch' # Note identification
beat_flag, beat_plot_flag, librosa_Flag = False, False, False # find bpm
lowpass_filter = True # apply lowpass filter to sound signal as its recorded
tone_generate = True # whether to generate a sine tone of recorded sound note, as a check of what freqs are identified

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

# create matplotlib figure and axes
fig, ax = plt.subplots(1, figsize=(15, 7))
# variable for plotting
x = fftfreq(pya_chunk,1/fs)
x = x[0:int(pya_chunk/2)]

# create a semilog line object initialised with random data for x-axis freq
line, = ax.semilogx(x, np.random.rand(int(pya_chunk/2)), '-', lw=2)

# basic formatting for the axes
ax.set_title('Live PSD waveform')
ax.set_xlabel('Log Freq Hz')
ax.set_ylabel('Log Amplitude')
ax.set_ylim(0, 10)
ax.set_xlim(20,)

# show the plot
plt.show(block=False)

print('stream started')
# Initialise array of notes in time sequence for storage
t, t_note, freqs_notes = [], [], []

start_t = time.time()
while int(time.time() - start_t) < seconds:
    data = stream.read(pya_chunk)
    # convert data to numpy array. Transient recording is of 1 second with pya_chunk number of array elements
    recording_transient = np.frombuffer(data, dtype=np.int16)

    # apply low-pass filter to remove high freq noise
    if lowpass_filter:
        B, A = signal.butter(3, 0.05, output='ba')
        recording_transient = signal.filtfilt(B, A, recording_transient)

    # find FFT and PSD and mornalise as per https://stackoverflow.com/questions/20165193/fft-normalization
    fft_transient = fft(recording_transient)/pya_chunk
    fft_transient = fft_transient[0:int(pya_chunk/2)]
    psd = (np.abs(fft_transient)**2)/(fs/pya_chunk)
    psd[1:-1] = 2*psd[1:-1]

    line.set_ydata(np.log10(psd))

    if freq_flag:
        freqs = freq.freq_identify(recording_transient, pya_channels, fs, method=freq_method, plot_flag=freq_plot_flag)
        notes = note.note_identify(freqs)
        t = np.append(t,[time.time()-start_t])
        t_note = np.append(t_note,[notes])
        freqs_notes = np.append(freqs_notes,[freqs])
        print(str(freqs) + ' , ' + notes)

    if beat_flag:
        bpm = beat.beat_detect(recording_transient, pya_channels, fs, filter_flag=1, plot_flag=beat_plot_flag, librosa_flag = librosa_flag)
        print(str(bpm))

    # update figure canvas
    fig.canvas.draw()
    fig.canvas.flush_events()

stream.stop_stream()
stream.close()
p.terminate()

print('stream stopped after {} seconds'.format(seconds))
notes_list = pd.DataFrame(np.vstack((t,t_note,freqs_notes)).T,columns='time note freqs'.split())

# Generate a sine tone of recorded sound note, as a check of what freqs are identified
if tone_generate:
    input("Press Enter to continue to play generated tone sound to simulate audio signal played")
    # Using script for tone generation in https://realpython.com/playing-and-recording-sound-python using simpleaudioo
    tone_signal = []
    for i in range(1,len(notes_list)):
        tone_signal = np.append(tone_signal, note.note_generator(float(notes_list.iloc[i,2]), fs, float(notes_list.iloc[i,0])-float(notes_list.iloc[i-1,0])))

    # Ensure that highest value is in 16-bit range
    tone_signal = tone_signal * (2 ** 15 - 1) / np.max(np.abs(tone_signal))
    # Convert to 16-bit data
    tone_signal = tone_signal.astype(np.int16)
    # Start playback
    play_obj = sa.play_buffer(tone_signal, 1, 2, fs)
    # Wait for playback to finish before exiting
    play_obj.wait_done()