import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import librosa

def beat_detect(myrecording, channels, fs, filter_flag=1, plot_flag=1, librosa_flag=False):

    if not librosa_flag:
        max_bpm = 1000 # Upper limit on what can be detected thru this script

        if channels > 1:
            rec_mono = np.mean(myrecording, axis=1)
        else:
            rec_mono = myrecording

        # Low pass filter on the signal to remove noise
        if filter_flag:
            B, A = signal.butter(3, 0.05, output='ba')
            rec_mono = signal.filtfilt(B,A, rec_mono)

        # Cross-correlation
        corr_rec = signal.correlate(rec_mono,rec_mono)

        # We're interested in only half of this correlated signal
        corr_rec = corr_rec[int(corr_rec.shape[0]/2)+1:]

        # Find the peaks from the cross-correlated signal
        p_ht = 0.1 * np.max(corr_rec)
        dist_min = (60/max_bpm)*fs
        corr_peaks, _ = scipy.signal.find_peaks(corr_rec,height=p_ht, distance=int(dist_min))
        err_flag = 0
        # Use time separation between the zero-peak and first peak to calculate bpm
        try:
            bpm_calc = 60/(np.diff(corr_peaks)[0]/fs)
            pass
        except:
            err_flag = 1
            bpm_calc = 0

        if plot_flag:
            # Time axis
            t = np.linspace(0,corr_rec.shape[0]-1,corr_rec.shape[0])/fs

            fig, axes = plt.subplots(1,squeeze=False)
            plt.tight_layout()

            axes[0,0].plot(t,corr_rec,'b')
            for p in corr_peaks:
                x_p = np.array([p/fs,p/fs])
                y_p = np.array([-1*np.max(corr_rec), np.max(corr_rec)])
                axes[0,0].plot(x_p,y_p,'-r') # Mark peaks detected by algo on plot
            axes[0,0].set_xlabel('t sec')
            axes[0,0].set_ylabel('Corr Coef')
            if ~err_flag:
                axes[0,0].set_title('Calculated BPM: {}'.format(int(bpm_calc)))
            else:
                axes[0,0].set_title('BPM could not be found')
    else:
        # Using Librosa
        bpm_calc, _ = librosa.beat.beat_track(rec_mono, sr=fs, start_bpm=120)

    return bpm_calc

