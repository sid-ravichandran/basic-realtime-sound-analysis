import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

def freq_identify(myrecording, channels, fs, method='welch', plot_flag=1):

    if channels > 1:
        rec_mono = np.mean(myrecording, axis=1)
    else:
        rec_mono = myrecording

    # Calculate PSD
    if method == 'welch':
        f_psd, psd = signal.welch(rec_mono, fs, nperseg=1024)
    else:
        f_psd, psd = signal.periodogram(rec_mono, fs)

    # Find the peaks from the psd
    p_ht = 0.2 * np.max(psd)
    dist_min = np.max([10/((f_psd.max() - f_psd.min())/f_psd.shape[0]),1]) # 10 Hz min separation between peaks
    corr_peaks, corr_peak_props = scipy.signal.find_peaks(psd, height=p_ht, distance=int(dist_min))

    if plot_flag:
        fig, axes = plt.subplots(1,squeeze=False)

        axes[0,0].plot(f_psd, psd)
        axes[0,0].set_xlabel('Freq Hz')
        axes[0,0].set_xlim([0, 1000])
        axes[0,0].set_ylabel('[V**2/Hz]')
        axes[0,0].set_title('PSD ' + method)

        plt.tight_layout()

    freqs = f_psd[corr_peaks]
    max_peak = np.argmax(corr_peak_props['peak_heights'])
    freq_max = freqs[max_peak]

    return freq_max
