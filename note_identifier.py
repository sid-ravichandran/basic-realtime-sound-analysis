import numpy as np
import pandas as pd

def note_identify(freqs):
    # Load freq map table
    freq_map = pd.read_pickle('freq_map.pkl')
    # n = freqs.shape[0] # no. of frequencies to extract notes for
    # notes = []

    arg = np.argmin(np.abs(freq_map['Freq'] - freqs))
    notes = freq_map['Note'][arg]

    # for i in range(n):
    #     arg = np.argmin(np.abs(freq_map['Freq'] - freqs[i]))
    #     notes.append(freq_map['Note'][arg])

    return notes

def note_generator(freq, fs, time):
    # https://realpython.com/playing-and-recording-sound-python/
    # Generate array with time*fs steps, ranging between 0 and seconds
    t = np.linspace(0, time, int(time * fs), False)

    # Generate a sine wave
    tone_signal = np.sin(freq * t * 2 * np.pi)
    return tone_signal

def key_identifier(notes):
    # Identify common keys based on notes provided
    raw_notes = []
    for i in range(len(notes)):
        note = notes[i]
        for n in range(len(note)):
            if type(note[n]) is int:
                break
        raw_notes.append(note[:n])

    # Check over major and minor scales
    maj = pd.read_pickle('major_scale.pkl')
    min = pd.read_pickle('minor_scale.pkl')

    maj_check = np.zeros([len(maj.iloc[0]),])
    min_check = np.zeros([len(min.iloc[0]),])
    tmp_maj_check = np.zeros([len(maj.iloc[0]),len(raw_notes)])
    for i in range(len(maj.iloc[0])):
        for j in range(len(raw_notes)):
            count = 0
            for k in range(1,len(maj.columns)):
                if raw_notes[j] == maj.iloc[i,k]:
                    count += 1
            tmp_maj_check[i,j] = count
    maj_check = np.sum(tmp_maj_check,axis=1)

    tmp_min_check = np.zeros([len(min.iloc[0]),len(raw_notes)])
    for i in range(len(min.iloc[0])):
        for j in range(len(raw_notes)):
            count = 0
            for k in range(1,len(min.columns)):
                if raw_notes[j] == min.iloc[i,k]:
                    count += 1
            tmp_min_check[i,j] = count
    min_check = np.sum(tmp_min_check,axis=1)

    major_key = maj.iloc[np.argmax(maj_check),0]
    minor_key = min.iloc[np.argmax(min_check),0]

    return major_key, minor_key

def scale_constructor():
    # construct major, minor scales
    base_notes = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']
    scale_notes = 7
    major_scale = pd.DataFrame(base_notes, index=base_notes, columns=['0'])
    minor_scale = pd.DataFrame(base_notes, index=base_notes, columns=['0'])
    l = len(base_notes)
    circ_indices = np.arange(0,l)
    circ_indices = np.append(circ_indices,circ_indices)

    # Construct Major scale  W-W-H-W-W-W-H, minor scale W-H-W-W-H-W-W
    major_step = np.array([2,4,5,7,9,11,12])
    minor_step = np.array([2,3,5,7,8,10,12])
    for i in range(l):
        for j in range(scale_notes):
            tmp_major_indices = circ_indices[major_step[j]:l+major_step[j]]
            tmp_minor_indices = circ_indices[minor_step[j]:l+minor_step[j]]
            tmp_major_stepped_notes = []
            tmp_minor_stepped_notes = []
            for k in range(l):
                tmp_major_stepped_notes.append(base_notes[tmp_major_indices[k]])
                tmp_minor_stepped_notes.append(base_notes[tmp_minor_indices[k]])
            major_scale[str(j+1)] = tmp_major_stepped_notes
            minor_scale[str(j+1)] = tmp_minor_stepped_notes

    major_scale.to_pickle('major_scale.pkl')
    minor_scale.to_pickle('minor_scale.pkl')



