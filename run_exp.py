import sys

# setting path
sys.path.append('..')
sys.path.append('../..')
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import torch
import warnings

warnings.filterwarnings("ignore")

import utils as u
from fun.gp_hsh import GP_HSH, CustomSquareWarpedGP
from fun.grid_ops import get_grid_idcs, gen_reg_grid


data_path = 'data/'

# data specs
n_max_samples = 50

sweep_params = {'acquisition_function': ['random', 'TS', 'KG', 'MVE', 'EI', 'UCB'],
                'n_init': [0,10],
                'seed': np.random.choice(np.arange(100),100, replace=False),
                'subj': ['002','003','004','006','007','008', '009', '010']
}

params = u.set_params(sweep_params=sweep_params)

search_r = 30
a_res = 10
s_res = 2

# generate grid
grid = gen_reg_grid(search_r,s_res,a_res)
pos = grid[:,1:]/search_r
rads  = (grid[:,0]-90)/90 
locs = np.hstack((pos, rads.reshape(-1,1)))


# init result array
result = pd.DataFrame(columns=['n_samples', 'subj', 'acquisition_function', 'n_init', 'mep_gen_seed','dist_loc', 'dist_ang', 'nmse'])
for j in tqdm(range(len(params))):
    # load data
    acquisition_function = params.loc[j,'acquisition_function']
    n_init = params.loc[j,'n_init']
    seed = params.loc[j,'seed']
    subj = params.loc[j, 'subj']

    res = np.load(os.path.join('data', 'result_'+subj+'.npy'), allow_pickle=True)[()]
    meps_gt = res['meps']
    grid = res['grid']

    # normalize to [-1,1]
    pos = grid[:,1:]/search_r
    rads  = (grid[:,0]-90)/90
    locs_gt = np.hstack((pos, rads.reshape(-1,1)))

    # convert to torch tensors
    train_Y = torch.tensor(meps_gt, dtype=torch.float32).reshape(-1,1)
    train_X = torch.tensor(locs_gt, dtype=torch.float32)
    locs_torch = torch.tensor(locs, dtype=torch.float32)

    # train ground truth model
    gt_model = CustomSquareWarpedGP(train_X, train_Y)
    gt_model.fit(n_restarts=5)

    # initiliaze optimization
    if n_init:
        grid_idcs = get_grid_idcs(locs, num_points=n_init, method='kmeans', seed=seed)
    else: 
        grid_idcs = []

    optimization = GP_HSH(all_locs=locs_torch, acquisition_function=acquisition_function)

    # run optimization
    for i in range(n_max_samples):
        if i<len(grid_idcs):
            next_state=torch.tensor(grid_idcs[i])
        else:
            next_state = optimization.sample_state()
        return_value = gt_model.posterior(locs_torch[next_state:next_state+1]).sample()[0,:,0]**2
 
        optimization.update(next_state, return_value)

        # save results
        if ((i+1)%2) == 0:
            nmse, dist_loc, dist_ang = optimization.get_nrmse_and_dist(gt_model, percentile=90)
            result.loc[len(result)] = {'n_samples': i+1,
                                'acquisition_function': acquisition_function,
                                'subj': subj,
                                'n_init': n_init,
                                'mep_gen_seed': seed,
                                'dist_loc': dist_loc,
                                'dist_ang': dist_ang,
                                'nmse': nmse}
            
result.to_pickle(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hs_hunting_exp.pkl'))