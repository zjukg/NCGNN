import numpy as np
import pandas as pd
import os

def save_main_results(model, dataset, n_step, use_extra_data, test_error, filename='main_results', full_args=''):
    file = 'results/%s.csv' % filename
    if not os.path.exists(file):
        df = pd.DataFrame({'model':[model],
        'dataset':[dataset,],
        'n_step':[n_step,],
        'use_extra_data':[use_extra_data,],
        'test_error':[test_error,],
        'full_args':[full_args,]})
        df.to_csv(file, index=False)
    else:
        df = pd.read_csv(file, index_col=False)
        new_row = pd.DataFrame({'model':[model],
        'dataset':[dataset,],
        'n_step':[n_step,],
        'use_extra_data':[use_extra_data,],
        'test_error':[test_error,],
        'full_args':[full_args,]})
        new_df = pd.concat([df, new_row], ignore_index=True)
        new_df.to_csv(file, index=False)
    return