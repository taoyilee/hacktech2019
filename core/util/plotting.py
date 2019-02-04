import wfdb
from matplotlib.backends.backend_pdf import PdfPages

from core.dataset.ltstdb_hea import LtstdbHea
import matplotlib.pyplot as plt
from biosppy.signals import ecg
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_hea(hea: LtstdbHea, output_dir="plot", samples=None):
    plt.figure(figsize=(20, 6))
    for i, s in enumerate(hea.signals):
        plt.plot(hea.timestamps[:samples], s[:samples], label=f"{hea.signal_spec[i].signal_description}")
    plt.xlabel("Time (s)")
    plt.ylabel(f"ECG Signal ({hea.signal_spec[0].adc_units})")
    plt.title(f"{hea.name} ECG Signals")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"{hea.name}_signals.png"))
    plt.close()


def plot_hea_biosppy(hea: LtstdbHea, output_dir="plot", samples=None):
    ts, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr = [], [], [], [], [], [], []
    for i, s in enumerate(hea.signals):
        descr = hea.signal_spec[i].signal_description
        tsi, filteredi, rpeaksi, ts_tmpli, templatesi, ts_hri, hri = ecg.ecg(signal=hea.signals[i][:samples],
                                                                             sampling_rate=hea.sampling_freq,
                                                                             show=False)
        ts.append(tsi), filtered.append(filteredi), rpeaks.append(rpeaksi), ts_tmpl.append(ts_tmpli)
        templates.append(templatesi), ts_hr.append(ts_hri), hr.append(hri)
        plt.figure(1)
        plt.plot(ts_hri, hri, 's-', label=f"{descr}")

        plt.figure()
        for t in templatesi:
            plt.plot(ts_tmpli, t, '-')
        plt.xlabel("Time (s)")
        plt.ylabel(f"ECG Signal ({hea.signal_spec[0].adc_units})")
        plt.title(f"{hea.name} Templates - {descr}")
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"{hea.name}_{descr}_templates.png"))

    plt.figure(1)
    plt.xlabel("Time (s)")
    plt.ylabel(f"HR (bpm)")
    plt.title(f"{hea.name} Hear Rate")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{hea.name}_hr.png"))
    plt.close("all")
    return ts, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr


def plotecg(x, y, begin=None, end=None, fs=250):
    if begin is None:
        begin = 0
    if end is None:
        end = len(x)
    time = (1 / fs) * np.arange(begin, end)
    plt.figure(1, figsize=(11.69, 8.27))
    plt.plot(time, x[begin:end, 0])
    plt.plot(time, x[begin:end, 1])
    cmap = LinearSegmentedColormap.from_list("ecg_annotation_cmap", ["r", "g", "b", "c", "y", "k", "m"])
    for i in range(7):
        plt.plot(time, - 1 + 0.1 * (y[begin:end, i] + 1.5 * i), color=cmap(i / 6))
    plt.grid()
    plt.ylim([-1, 1])


def plot_annotation(y, begin=None, end=None, fs=250):
    if begin is None:
        begin = 0
    if end is None:
        end = len(y)
    time = (1 / fs) * np.arange(begin, end)
    cmap = LinearSegmentedColormap.from_list("ecg_annotation_cmap", ["r", "g", "b", "c", "y", "k", "m"])
    for i in range(7):
        plt.plot(time, - 1 + 0.1 * (np.array(y[begin:end] == i, dtype=np.int) + 1.5 * i), linestyle="--",
                 color=cmap(i / 6))


def plotecg_validation(x, y_true, y_pred, begin, end):
    # helper to plot ecg
    plt.figure(1, figsize=(11.69, 8.27))
    plt.subplot(211)
    plt.plot(x[begin:end, 0])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 0])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 1])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 2])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 3])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 4])
    plt.subplot(211)
    plt.plot(y_pred[begin:end, 5])

    plt.subplot(212)
    plt.plot(x[begin:end, 1])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 0])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 1])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 2])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 3])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 4])
    plt.subplot(212)
    plt.plot(y_true[begin:end, 5])


class RecordPlotter:
    def __init__(self, signal_length_sec, ecg_database, n_pages=10):
        self.signal_length_sec = signal_length_sec
        self.ecg_database = ecg_database
        self.n_pages = n_pages

    def __call__(self, record_hea):
        record_name = os.path.splitext(record_hea)[0]
        print(f"processing record {record_name}")
        record = wfdb.io.rdrecord(record_name)
        pdf_out = os.path.join(f'plot/{self.ecg_database}_{record.record_name}_view.pdf')
        if os.path.isfile(pdf_out):
            return
        print(f"{record.p_signal.shape}")
        print(f"{record.sig_name}")
        print(f"{record.sig_len}")
        samples_per_segment = int(self.signal_length_sec / (1 / record.fs))
        with PdfPages(pdf_out) as pdf:
            for starting_idx in range(0, min(self.n_pages * samples_per_segment, record.sig_len), samples_per_segment):
                subslice = slice(starting_idx, min(starting_idx + samples_per_segment, record.sig_len))
                plt.figure(1, figsize=(11.69, 8.27))

                time = (1 / record.fs) * np.arange(subslice.start, subslice.stop)
                colors = ["red", "blue"]
                for i in range(record.n_sig):
                    plt.subplot(record.n_sig + 1, 1, i + 1)
                    plt.plot(time, record.p_signal[subslice, i], label=f"{record.sig_name[i]}", color=colors[i])
                    plt.ylabel("Voltage (mV)")
                    plt.title(f"{self.ecg_database} Record #{record.record_name} {subslice.start}-{subslice.stop}")
                    plt.xticks(np.arange(min(time), max(time) + 1, 1))
                    plt.yticks(np.arange(-1.5, 2, 0.5))
                    for xx in np.arange(min(time), max(time), 0.2):
                        plt.vlines(xx, -1.5, 1.5, color="gray", linewidth=0.5)
                    for xx in np.arange(min(time), max(time), 0.04):
                        plt.vlines(xx, -1.5, 1.5, color="gray", linewidth=0.1)
                    for yy in np.arange(-1.5, 1.5, 0.1):
                        plt.hlines(yy, min(time), max(time), color="gray", linewidth=0.1)

                    plt.xlim([min(time), max(time)])
                    plt.ylim([-1.5, 1.5])
                    plt.grid(which='both')
                    plt.legend(loc="upper right")
                    plt.xlabel("Time (s)")
                plt.text(time[10], -1.5,
                         f"fs = {record.fs} Hz signal_total_len = {record.sig_len / record.fs / 60:.2f} min. {samples_per_segment} pts in this plot")
                for i, cmt in enumerate(record.comments):
                    plt.text(time[10], -1.3 + 0.2 * i, cmt)
                plt.subplot(record.n_sig + 1, 1, record.n_sig + 1)
                fft_pts = 2048
                for i in range(record.n_sig):
                    spectrum = np.fft.fft(record.p_signal[subslice, i], n=fft_pts)
                    freq = np.fft.fftfreq(n=fft_pts, d=1 / record.fs)
                    spectrum = spectrum[10:int(fft_pts / 2)]
                    freq = freq[10:int(fft_pts / 2)]
                    spectrum = abs(spectrum) ** 2
                    spectrum /= np.max(spectrum)
                    plt.plot(freq, spectrum, label=f"{record.sig_name[i]}", color=colors[i])
                plt.grid()
                plt.xlim([1, 75])
                plt.title(f"{self.ecg_database} Record #{record.record_name} Spectrum")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Relative Power")
                plt.legend(loc="upper right")
                plt.tight_layout()
                pdf.savefig()
                plt.close()
