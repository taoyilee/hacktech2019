# Explainable ECG
This is a UCI Computer Science course project which aims to build explainable AI models to classify pathological events in electrocardiograms (ECG).

## System Requirements
1. Python 3.6
2. Keras==2.2.4
3. tensorflow-gpu==1.12.0

## Quick Start
1. Setup [Python virtual environment](https://virtualenv.pypa.io/en/latest/userguide/#usage) ```virtualenv ENV```, where ENV is a directory to place the new virtual environment
2. Install requirments ```pip  install -r requirements.txt```

## Configuration File
```ini
[DEFAULT]
loglevel = DEBUG
logdir = log
experiments_dir = experiments

[nsrdb]
; Change this path to the folder which you store nsrdb
dataset_path = C:\.....\Dataset\nsrdb

[mitdb]
; Change this path to the folder which you store mitdb
dataset_path = C:\.....\Dataset\mitdb

[preprocessing]
NSR_DB_TAG = 0
MIT_DB_TAG = 1

; Take 2 records from MIT_DB/NSR_DB to build dev set
dev_record_each = 2
; Take 5 records from MIT_DB/NSR_DB to build test set
test_record_each = 5

batch_size = 512
sequence_length = 1300
overlap_percent = 5
augmentation = True
random_time_scale_percent = 20
dilation_factor = 1
awgn_rms_percent = 2

[RNN-train]
rnn_output_features = 32
l2_regularization = 0
dropout = 0.2
initial_lr = 0.001
initial_weights = None
model_output = trained_model/model.h5
tensorboard_dir = tensorboard
epochs = 100
patientce_reduce_lr = 2
early_stop = True
verbosity = 1

[RNN-test]
test_set = experiments/7Xa9GShsje_0122_112200/test.pickle
model_json = experiments/7Xa9GShsje_0122_112200/model.json
weights = experiments/7Xa9GShsje_0122_112200/final_weights.h5
batch_size = 32
```
## Acknowledgement
The authors of this software package would like to thank following authors and their efforts in setting up 
long-term S-T database.
    
> Franc Jager, Alessandro Taddei, George B. Moody, Michele Emdin, Gorazd Antolic, 
> Roman Dorn, Ales Smrdel, Carlo Marchesi, and Roger G. Mark. Long-term ST database: 
> a reference for the development and evaluation of automated ischaemia detectors and 
> for the study of the dynamics of myocardial ischaemia. Medical & Biological Engineering & 
> Computing 41(2):172-183 (2003).

And Physionet for hosting this dataset:        
> Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB,
> Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research
> Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 
> [http://circ.ahajournals.org/content/101/23/e215.full](); 2000 (June 13). 

## License 

Copyright 2018 Tao-Yi Lee, Kenneth Stewart and Saehanseul Yi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
