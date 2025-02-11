import numpy as np
import pandas as pd

def set_params(single_params={}, sweep_params={}, default_params={}):

    if default_params:
        params = pd.DataFrame.from_dict(default_params, orient='index').T

        for single_param in single_params.keys():
            new_params = default_params.copy()

            for p in single_params[single_param]:
                new_params[single_param] = p
                
                params.loc[len(params)] = new_params
    
    else:
        params = pd.DataFrame([np.zeros(len(sweep_params.keys()))], columns=sweep_params.keys())

    for sweep_param in sweep_params.keys():
        new_params = params.copy()
        params_ = None

        for p in sweep_params[sweep_param]:
            new_params[sweep_param] = p
            
            params = pd.concat([params_, new_params], ignore_index=True)
            params_ = params

    return params


def cog(x, y, threshold=None, percentile=None):
    if threshold is None:
        if percentile is None:
            threshold = -np.inf
        else:
            threshold = np.percentile(y,percentile)
    x_ = x[y>threshold]
    y_ = y[y>threshold]

    cog = (x_*y_.reshape(-1,1)).sum(0) / y_.sum()
    return cog




