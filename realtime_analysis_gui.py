# SIMILAR TO SOUND_REALTIME BUT USING TKINTER GUI
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
from scipy import signal
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import pandas as pd

import Beat_detector as beat
import Freq_identifier as freq
import note_identifier as note

freq_flag, freq_plot_flag, freq_method = True, False, 'periodogram' # Note identification
beat_flag, beat_plot_flag, librosa_Flag = False, False, False # find bpm
lowpass_filter = True # apply lowpass filter to sound signal as its recorded
store_notes = False

pya_chunk = 1024 # PyAudio recording chunk
pya_format = pyaudio.paInt16
pya_channels = 1 # single channel recording
fs = 44100  # Sample rate
seconds = 10  # Duration of recording

p = pyaudio.PyAudio()
stream = p.open(
    format = pya_format,
    channels = pya_channels,
    rate = fs,
    input = True,
    output = True,
    frames_per_buffer = pya_chunk
)

# Creating tkinter main window
win = tk.Tk()
win.title("Real Time Sound Data")
# Title Label
ttk.Label(win,
          text = "Real Time Sound Data",
          font = ("Times New Roman", 15),
          background = 'green',
          foreground = "white").grid(column = 0,
                                     row = 0)

# Creating scrolled text
# area widget
text_area = scrolledtext.ScrolledText(win,
                                      wrap=tk.WORD,
                                      width=50,
                                      height=1,
                                      font=("Times New Roman",
                                            15))

text_area.grid(column=0, pady=10, padx=10)

# create matplotlib figure and axes
fig, ax = plt.subplots(1, figsize=(10, 4))
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
if store_notes:
    # Initialise array of notes in time sequence for storage
    t, t_note, freqs_notes = [], [], []

text_area.insert(tk.INSERT, 'START' + '\n')

print('stream started')
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

    freqs = freq.freq_identify(recording_transient, pya_channels, fs, method=freq_method, plot_flag=False)
    notes = note.note_identify(freqs)
    if store_notes:
        t = np.append(t, [time.time() - start_t])
        t_note = np.append(t_note, [notes])
        freqs_notes = np.append(freqs_notes, [freqs])
    # text_area.delete('1.0', tk.END)
    # win.after(1000)
    text_area.insert(tk.INSERT, 't = ' + str(round(time.time() - start_t)) + ' sec Freq ' + str(round(freqs,2)) + ' Hz , Note ' + notes + '\n')
    text_area.see(tk.END)
    text_area.update()

    if beat_flag:
        bpm = beat.beat_detect(recording_transient, pya_channels, fs, filter_flag=1, plot_flag=beat_plot_flag, librosa_flag = librosa_flag)
        print(str(bpm))

    # # update figure canvas
    fig.canvas.draw()
    fig.canvas.flush_events()

print('stream stopped after {} seconds'.format(seconds))
if store_notes:
    notes_list = pd.DataFrame(np.vstack((t,t_note,freqs_notes)).T,columns='time note freqs'.split())

stream.stop_stream()
stream.close()
p.terminate()

# Making the text read only
# text_area.configure(state ='disabled')
win.mainloop()
win.quit()