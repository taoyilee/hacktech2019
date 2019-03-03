import wfdb
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from biosppy.signals.ecg import hamilton_segmenter


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
        print("processing record", record_name)
        record = wfdb.io.rdrecord(record_name)
        pdf_out = os.path.join(f'plot/{self.ecg_database}_{record.record_name}_view.pdf')
        if os.path.isfile(pdf_out):
            return
        print(record.p_signal.shape)
        print(record.sig_name)
        print(record.sig_len)
        samples_per_segment = int(self.signal_length_sec / (1 / record.fs))
        with PdfPages(pdf_out) as pdf:
            for starting_idx in range(0, min(self.n_pages * samples_per_segment, record.sig_len), samples_per_segment):
                subslice = slice(starting_idx, min(starting_idx + samples_per_segment, record.sig_len))
                plt.figure(1, figsize=(11.69, 8.27))

                time = (1 / record.fs) * np.arange(subslice.start, subslice.stop)
                colors = ["red", "blue"]
                for i in range(record.n_sig):
                    plt.subplot(record.n_sig + 1, 1, i + 1)
                    plt.plot(time, record.p_signal[subslice, i], label=record.sig_name[i], color=colors[i])

                    plt.ylabel("Voltage (mV)")
                    plt.title(
                        "%s Record #%s %d-%d" % (self.ecg_database, record.record_name, subslice.start, subslice.stop))
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
                         "fs = %d Hz signal_total_len = %.2f min. %d pts in this plot" % (
                         record.fs, record.sig_len / record.fs / 60, samples_per_segment))
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


class RecordPNGPlotter:
    def __init__(self, signal_length_sec, ecg_database):
        self.signal_length_sec = signal_length_sec
        self.ecg_database = ecg_database

    def __call__(self, record_hea):
        record_name = os.path.splitext(record_hea)[0]
        print(f"processing record {record_name}")
        record = wfdb.io.rdrecord(record_name)
        samples_per_segment = int(self.signal_length_sec / (1 / record.fs))

        for starting_idx in range(0, record.sig_len, samples_per_segment):

            # if os.path.isfile(img_out):
            #     continue
            subslice = slice(starting_idx, min(starting_idx + samples_per_segment, record.sig_len))
            plt.figure()
            time = (1 / record.fs) * np.arange(subslice.start, subslice.stop)
            colors = ["red", "blue"]
            arrythmia = False
            for i in range(record.n_sig):
                signal = record.p_signal[subslice, i]
                plt.plot(time, signal, label=f"{record.sig_name[i]}", color=colors[i])
                out = ecg.ecg(signal=signal, sampling_rate=record.fs, show=False)
                plt.scatter(time[out["rpeaks"]], signal[out["rpeaks"]])
                rr_int = np.array(time[out["rpeaks"][1:]] - time[out["rpeaks"][:-1]])
                for rp_idx_a, rp_idx_b, rr in zip(out["rpeaks"][:-1], out["rpeaks"][1:], rr_int):
                    plt.hlines(signal[rp_idx_a], time[rp_idx_a], time[rp_idx_b])
                    plt.text((time[rp_idx_a] + time[rp_idx_b]) / 2, signal[rp_idx_a], f"{rr:.2f}")
                hrv_perc = np.round(100 * (rr_int / np.min(rr_int) - 1), 2)
                if np.any(hrv_perc > 15):
                    arrythmia = True
                plt.text(min(time), -1.5 + 0.5 * i, hrv_perc)
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
            if arrythmia:
                img_out = os.path.join("plot", "arrythmia",
                                       f'{self.ecg_database}_{record.record_name}_{starting_idx:05d}_view.png')
            else:
                img_out = os.path.join("plot", "nsr",
                                       f'{self.ecg_database}_{record.record_name}_{starting_idx:05d}_view.png')
            plt.savefig(img_out)
            plt.close()


class RecordPNGPlotterDF:
    def __init__(self, mitdb_path, signal_length_sec, ecg_database):
        self.mitdb_path = mitdb_path
        self.signal_length_sec = signal_length_sec
        self.ecg_database = ecg_database

    def __call__(self, data_frame):
        data_frame = data_frame[1]
        record_path = os.path.join(self.mitdb_path, f"{data_frame['Record']}")
        print(f"processing record {record_path}")
        record = wfdb.io.rdrecord(record_path)
        starting_idx = data_frame["Start_Index"]
        ending_idx = data_frame["End_Index"]
        subslice = slice(starting_idx, ending_idx)
        plt.figure()
        time = (1 / record.fs) * np.arange(subslice.start, subslice.stop)
        colors = ["red", "blue"]
        for i in range(record.n_sig):
            signal = record.p_signal[subslice, i]
            plt.plot(time, signal, label=f"{record.sig_name[i]}", color=colors[i])
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
        img_out = os.path.join("plot", "pending_review",
                               f'{self.ecg_database}_{record.record_name}_{starting_idx:05d}_view.png')
        plt.savefig(img_out)
        plt.close()


def process_slice(record, subslice, threshold=15):
    time = (1 / record.fs) * np.arange(subslice.start, subslice.stop)

    for i in range(record.n_sig):
        signal = record.p_signal[subslice, i]
        rpeaks, = hamilton_segmenter(signal, record.fs)
        # out = ecg.ecg(signal=signal, sampling_rate=record.fs, show=False)
        try:
            rr_int = np.vstack((rr_int, np.array(time[rpeaks[1:]] - time[rpeaks[:-1]])))
        except Exception:
            rr_int = np.array(time[rpeaks[1:]] - time[rpeaks[:-1]])
    rr_int = np.mean(rr_int, axis=0)
    hrv_perc = np.abs(np.round(100 * (rr_int / np.min(rr_int) - 1), 2))
    arrythmia = np.any(hrv_perc > threshold)
    return record.record_name, subslice.start, subslice.stop, arrythmia, rr_int.max(), rr_int.min(), hrv_perc.max(), hrv_perc.min(), hrv_perc, rr_int


class RecordRR:
    def __init__(self, signal_length_sec, ecg_database):
        self.signal_length_sec = signal_length_sec
        self.ecg_database = ecg_database

    def __call__(self, record_hea):
        record_name = os.path.splitext(record_hea)[0]
        print(f"processing record {record_name}")
        record = wfdb.io.rdrecord(record_name)
        samples_per_segment = int(self.signal_length_sec / (1 / record.fs))

        subslice = [slice(starting_idx, min(starting_idx + samples_per_segment, record.sig_len)) for starting_idx in
                    range(0, record.sig_len, samples_per_segment)]
        return tuple(map(lambda x: process_slice(record, x), subslice))
