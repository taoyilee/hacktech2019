
# Explainmyheart
Using this tool, you can load a model that predicts features of an electrocardiogram, and then it will also explain why the particular prediction was made!


```python
from keras.models import model_from_json
from keras.utils import plot_model

import lime
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.segmentation import mark_boundaries
import wfdb

import seaborn as sns
sns.set_style('ticks')
```


```python
MODEL_LOCATION = 'assets/model.json'
WEIGHTS_LOCATION = 'assets/final_weights.h5'

with open(MODEL_LOCATION, 'r') as f:
    model_json = f.read()

model_to_interpret = model_from_json(model_json)
model_to_interpret.load_weights(WEIGHTS_LOCATION)
```


```python
test_data_dir = input('Enter test data location:')
```

    Enter test data location:G:\Team Drives\Hacktech2019\mitdb
    


```python
rec = wfdb.rdrecord(os.path.join(test_data_dir, '102'))
wfdb.plot_wfdb(record=rec)
```


![png](assets/images/output_4_0.png)

