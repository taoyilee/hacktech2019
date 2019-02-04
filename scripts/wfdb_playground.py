import wfdb
from core.dataset.qtdb import extract_annotated_ecg
import matplotlib.pyplot as plt
import numpy as np

# ann = wfdb.rdann("data/qtdb/sel30", "man")
# print(ann.sample)
# print(ann.symbol)
# ann = wfdb.rdann("data/qtdb/sel30", "pu")
# print(ann.sample)
# print(ann.symbol)
# ann = wfdb.rdann("data/qtdb/sel30", "pu0")
# print(ann.sample)
# print(ann.symbol)
# ann = wfdb.rdann("data/qtdb/sel30", "pu1")
# print(ann.sample)
# print(ann.symbol)
# ann = wfdb.rdann("data/qtdb/sel30", "q1c")
# print(ann.sample)
# print(ann.symbol)
# ann = wfdb.rdann("data/qtdb/sel30", "qt1")
# print(ann.sample)
# print(ann.symbol)
# wfdb.show_ann_classes()
# wfdb.show_ann_labels()

record = wfdb.rdrecord("data/qtdb/sel30")
# print(record.p_signal.shape)
xx, yy = extract_annotated_ecg("data/qtdb/sel30")

plt.figure(1, figsize=(11.69, 8.27))
time = (1 / record.fs) * np.arange(0, len(xx))
plt.plot(time, xx[:, 0])
plt.plot(time, xx[:, 1])
for i, tag in enumerate(np.unique(yy)):
    plt.scatter(time[yy == tag], 0.1 + 1e-2 * i * np.ones_like(time[yy == tag]), s=1)

plt.xlim([0, 5])
plt.savefig("plot/sel30_xx.png")
