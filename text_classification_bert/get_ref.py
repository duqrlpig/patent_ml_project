import os
import pandas as pd

def get_ref_claim(FILE_DIR):

    test_df = pd.read_csv(FILE_DIR+"test.csv", header=None)
    test_df.head()