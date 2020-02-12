import pandas as pd 
import numpy as np

rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5)) 
ser
ser.mean()
ser.median()
ser.mode()





