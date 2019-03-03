import bokeh
from bokeh.plotting import figure, output_file
from bokeh.models import ColorBar
from bokeh.palettes import PiYG
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.transform import linear_cmap

import matplotlib.pyplot as plt
import numpy as np
import wfdb

def make_html_plot(signal: np.ndarray, explanations=None, title: str=''):
    # explanations = [np.array(explanation, dtype=np.dtype('int, float'))
    #                 for explanation in explanations.values()]
    
    indices = [np.array(elem[0], dtype=np.int) for elem in explanations.values()]
    importances = [np.array(elem[1], dtype=np.float) for elem in explanations.values()]

    p = figure()
    p.line(list(range(len(signal))), signal[:, 0])
    # p.circle(list(range(len(signal))), signal[:, 0])
    
    if  explanations is not None:
        p.scatter(indices[0], signal[indices[0]])

    return file_html(p, CDN, title)
