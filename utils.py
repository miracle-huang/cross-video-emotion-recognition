import numpy as np
import random
import tensorflow as tf
import os
import pandas as pd

def set_global_random_seed(random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)