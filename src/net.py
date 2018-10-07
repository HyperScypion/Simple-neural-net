import random
import pandas
import numpy as np 

class MLP(object):

def sigma(x):
    return (1 / (1 + np.exp(-x)))
