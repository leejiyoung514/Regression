from matplotlib import pyplot as plt
import mglearn
import pandas as pd
import numpy as np
from matplotlib import font_manager, rc
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import os

font_name=font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

os.environ["PATH"]+=os.pathsep+\
                    'C:/Program Files (x86)/Graphviz2.38/bin/'

mglearn.plots.plot_logistic_regression_graph()
mglearn.plots.plot_single_hidden_layer_graph()
mglearn.plots.plot_two_hidden_layer_graph()

