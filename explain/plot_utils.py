import bokeh
from bokeh.plotting import figure, output_file
from bokeh.resources import CDN
from bokeh.embed import file_html

import matplotlib.pyplot as plt
import numpy as np
import wfdb

def make_html_plot(signal: np.ndarray, importance: np.ndarray=None,
                   title: str=''):
    p = figure()
    p.line(list(range(len(signal))), signal[:, 0])

    return file_html(p, CDN, title)
