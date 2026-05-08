import numpy as np
import pandas as pd

class Code:
    def load_data(name):
        df = pd.read_csv(name, encoding='CP1251')
        return df
