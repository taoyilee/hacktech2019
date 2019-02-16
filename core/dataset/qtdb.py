import math
import os
from os.path import basename
import configparser as cp
import numpy as np
import wfdb
import logging

from core.dataset.preprocessing import ECGDataset, ECGTaggedPair
from core.dataset.helper import normalize_signal

logger = logging.getLogger('qtdb')
logger.setLevel(logging.DEBUG)
config = cp.ConfigParser()
config.read("config.ini.template")
REJECTED_TAGS = tuple(config["qtdb"].get("reject_tags").split(","))
try:
    os.makedirs(config["logging"].get("logdir"))
except FileExistsError:
    pass
fh = logging.FileHandler(os.path.join(config["logging"].get("logdir"), "qtdb.log"), mode="w+")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def is_valid_record(annotations: wfdb.Annotation, rejected_tags=REJECTED_TAGS):
    for tag in rejected_tags:
        if tag in annotations.symbol:
            logger.log(logging.WARN, f"Invalid tag {tag}")
            return False
    return True


def slice_annotated_interval(record_name, annotator='q1c') -> (wfdb.Record, wfdb.Annotation):
    logger.log(logging.DEBUG, f"Record name: {record_name}")
    annotation = wfdb.rdann(record_name, extension=annotator)
    logger.log(logging.DEBUG, f"Original annotation: {''.join(annotation.symbol)}")
    record = wfdb.rdrecord(record_name, sampfrom=annotation.sample[0], sampto=annotation.sample[-1])
    return record, annotation


def translate_annotations(annotation: wfdb.Annotation):
    symbol = np.array(annotation.symbol)
    p_indexes = np.arange(annotation.ann_len)[np.array([s == "p" for s in annotation.symbol])]
    if len(p_indexes) != 0:
        symbol[(p_indexes - 1)[np.array([s == "(" for s in symbol[p_indexes - 1]])]] = "a"
        symbol[(p_indexes + 1)[np.array([s == ")" for s in symbol[p_indexes + 1]])]] = "b"
        symbol[p_indexes] = "P"
    n_indexes = np.arange(annotation.ann_len)[np.array([s == "N" for s in annotation.symbol])]
    if len(n_indexes) != 0:
        symbol[(n_indexes - 1)[np.array([s == "(" for s in symbol[n_indexes - 1]])]] = "Q"
        symbol[n_indexes] = "R"
        symbol[(n_indexes + 1)[np.array([s == ")" for s in symbol[n_indexes + 1]])]] = "S"
    t_indexes = np.arange(annotation.ann_len)[np.array([s == "t" for s in annotation.symbol])]
    if len(t_indexes) != 0:
        symbol[(t_indexes - 1)[np.array([s == "(" for s in symbol[t_indexes - 1]])]] = "c"
        symbol[t_indexes] = "T"
        symbol[(t_indexes + 1)[np.array([s == ")" for s in symbol[t_indexes + 1]])]] = "d"
    u_indexes = np.arange(annotation.ann_len)[np.array([s == "u" for s in annotation.symbol])]
    if len(u_indexes) != 0:
        symbol[(u_indexes - 1)[np.array([s == "(" for s in symbol[u_indexes - 1]])]] = "e"
        symbol[u_indexes] = "U"
        symbol[(u_indexes + 1)[np.array([s == ")" for s in symbol[u_indexes + 1]])]] = "f"
    annotation.symbol = symbol.tolist()
    return annotation


def filter_u(annotation: wfdb.Annotation):  # filter out U intervals and associated parenthesises
    marked_out = np.array([s == "u" for s in annotation.symbol])
    u_indexes = np.arange(annotation.ann_len)[marked_out]
    if len(u_indexes) == 0:
        return annotation
    u_left_index = u_indexes - 1
    u_right_index = u_indexes + 1
    marked_out[u_left_index] = np.array([s == "(" for s in np.array(annotation.symbol)[u_left_index]])
    marked_out[u_right_index] = np.array([s == ")" for s in np.array(annotation.symbol)[u_right_index]])
    annotation.sample = np.array(annotation.sample)[np.invert(marked_out)]
    annotation.symbol = np.array(annotation.symbol)[np.invert(marked_out)].tolist()
    annotation.ann_len = len(annotation.sample)
    return annotation


def filter_t(annotation):  # filter out the left parenthesis associated with T
    t_indexes = np.arange(annotation.ann_len)[np.array([s == "t" for s in annotation.symbol])]
    if len(t_indexes) == 0:
        return annotation
    t_left_index = t_indexes - 1
    marked_out = np.array([False for _ in range(annotation.ann_len)])
    marked_out[t_left_index] = np.array([s == "(" for s in np.array(annotation.symbol)[t_left_index]])
    annotation.sample = np.array(annotation.sample)[np.invert(marked_out)]
    annotation.symbol = np.array(annotation.symbol)[np.invert(marked_out)].tolist()
    annotation.ann_len = len(annotation.sample)
    return annotation


def extract_annotated_ecg(record: wfdb.Record, annotation):
    logger.log(logging.INFO, f"record: {record.record_name}")
    logger.log(logging.INFO, f"annotations: {''.join(annotation.symbol)}")
    annotation_tags = np.empty(len(record.p_signal), dtype=object)
    for sample_left, sample_right, symbol_left, symbol_right in zip(annotation.sample[:-1] - annotation.sample[0],
                                                                    annotation.sample[1:] - annotation.sample[0],
                                                                    annotation.symbol[:-1], annotation.symbol[1:]):
        logger.log(logging.DEBUG, f"{symbol_left + symbol_right}")
        annotation_tags[sample_left:sample_right] = symbol_left + symbol_right
    return record.p_signal, annotation_tags


def one_hot_encode_label(label, valid_segments, categories):
    unique_categories = {}
    for c, t in zip(categories, valid_segments):
        unique_categories[c] = unique_categories.get(c, []) + [t]
    logger.log(logging.DEBUG, f"unique_categories: {unique_categories}")
    return np.array([[lbl in v for lbl in label] for _, v in unique_categories.items()], dtype=int).T


def split_sequence(x, n, o):
    # split seq; should be optimized so that remove_seq_gaps is not needed.
    upper = math.ceil(x.shape[0] / n) * n
    print("splitting on", n, "with overlap of ", o, "total datapoints:", x.shape[0], "; upper:", upper)
    for i in range(0, upper, n):
        # print(i)
        if i == 0:
            padded = np.zeros((o + n + o, x.shape[1]))  ## pad with 0's on init
            padded[o:, :x.shape[1]] = x[i:i + n + o, :]
            xpart = padded
        else:
            xpart = x[i - o:i + n + o, :]
        if xpart.shape[0] < i:
            padded = np.zeros((o + n + o, xpart.shape[1]))  ## pad with 0's on end of seq
            padded[:xpart.shape[0], :xpart.shape[1]] = xpart
            xpart = padded

        xpart = np.expand_dims(xpart, 0)  ## add one dimension; so that you get shape (samples,timesteps,features)
        try:
            xx = np.vstack((xx, xpart))
        except UnboundLocalError:  ## on init
            xx = xpart
    print("output: ", xx.shape)
    return xx


def remove_seq_gaps(x, y):
    # remove parts that are not annotated <- not ideal, but quickest for now.
    window = 150
    c = 0
    cutout = []
    include = []
    print("filterering.")
    print("before shape x,y", x.shape, y.shape)
    for i in range(y.shape[0]):
        c = c + 1
        if c < window:
            include.append(i)
        if sum(y[i, 0:5]) > 0:
            c = 0
        if c >= window:
            # print ('filtering')
            pass
    x, y = x[include, :], y[include, :]
    print(" after shape x,y", x.shape, y.shape)
    return x, y


def split_dataset(dataset_array, train_percent=92, dev_percent=3, test_percent=3):
    train_num = len(dataset_array) * train_percent // 100
    dev_num = len(dataset_array) * dev_percent // 100
    test_num = len(dataset_array) * test_percent // 100
    train_set = ECGDataset("train", dataset_array[:train_num])
    dev_set = ECGDataset("dev", dataset_array[train_num:train_num + dev_num])
    test_set = ECGDataset("test", dataset_array[train_num + dev_num:train_num + dev_num + test_num])
    logger.log(logging.INFO, f"Training set size: {len(train_set)}({100 * len(train_set) / len(dataset_array):.2f}%)")
    logger.log(logging.INFO, f"Dev set size: {len(dev_set)}({100 * len(dev_set) / len(dataset_array):.2f}%)")
    logger.log(logging.INFO, f"Testing set size: {len(test_set)}({100 * len(test_set) / len(dataset_array):.2f}%)")
    return train_set, dev_set, test_set


def load_dat(datfiles, valid_segments, categories, exclude=None, rejected_tags=REJECTED_TAGS):
    tagged_pair = []
    for datfile in datfiles:
        if exclude is not None and basename(datfile).split(".", 1)[0] in exclude:
            continue
        print(f"Processing {datfile}")
        record_name = os.path.splitext(datfile)[0]
        record, annotation = slice_annotated_interval(record_name, annotator='q1c')
        if not is_valid_record(annotation, rejected_tags):
            logger.log(logging.WARN, f"{record.record_name} skipped due to invalid tag")
            continue
        annotation = translate_annotations(annotation)
        # annotation = filter_u(annotation)
        # annotation = filter_t(annotation)
        x, y = extract_annotated_ecg(record, annotation)
        logger.log(logging.DEBUG, f"Shapes of (x, y) are {x.shape} {y.shape}")
        logger.log(logging.DEBUG, f"Segment encoded Y is {y[:10]}")
        y = one_hot_encode_label(y, valid_segments, categories)
        logger.log(logging.DEBUG, f"Shapes of (x, y) are {x.shape} {y.shape}")
        logger.log(logging.DEBUG, f"Categorical encoded Y is {y[:10, :]}")
        x = normalize_signal(x)
        tagged_pair.append(ECGTaggedPair(x, y, record.fs, record.record_name))
    logger.log(logging.DEBUG, f"tagged_pair has {len(tagged_pair)} pair of data")
    return tagged_pair
