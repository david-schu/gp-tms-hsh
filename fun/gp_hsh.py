import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import qKnowledgeGradient, analytic, qMaxValueEntropy
from botorch.models.transforms.outcome import Standardize
import utils as u

NUM_FANTASIES = 15
NUM_RESTARTS = 2
RAW_SAMPLES = 700
BOUNDS = torch.stack([-1*torch.ones(3), torch.ones(3)])

class CustomSquareWarpedGP(SingleTaskGP):
    def __init__(self, train_X, train_Y, **kwargs):
        train_Y = torch.sqrt(train_Y)
        super().__init__(train_X, train_Y, outcome_transform=Standardize(m=1), **kwargs)
    
    
    def predict(self, X: torch.Tensor, predict_raw: bool = False):
        self.eval()
        with torch.no_grad():
            posterior = self.posterior(X)
            mean = posterior.mean
            var = posterior.variance
        if predict_raw:
            return mean, var
        else:  
            return mean**2, mean**2 * var
    
    def fit(self, n_restarts=1):
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        self.train()
        mll.train()
        fit_gpytorch_model(mll, max_retries=n_restarts)



class GP_HSH:
    def __init__(self, all_locs, max_n_sample=1, acquisition_function='REJ', af_params=None):
        self.all_locs = all_locs
        self.sampled_loc_idcs = torch.empty(0, dtype=int)
        self.sampled_returns = torch.empty(0, dtype=int)
        self.gp_model = None
        self.acquisition_function = acquisition_function
        self.n_sampled = torch.zeros(len(all_locs))
        self.max_n_sample = max_n_sample
        self.af_params=af_params


    def sample_state(self):
        if len(self.sampled_loc_idcs) == 0:
            # Start by sampling random idc
            return torch.randint(len(self.all_locs),(1,))
        else:
            # Select the next state using the specified acquisition function
            next_state = self._acquisition_selection()
            return next_state

    def update(self, state, return_value, n_restarts=1):
        # Update the sampled states and returns
        self.sampled_loc_idcs = torch.cat((self.sampled_loc_idcs, state.reshape(1)))
        self.sampled_returns = torch.cat((self.sampled_returns, return_value))
        self.n_sampled[state] +=1
        self.available_loc_idcs = torch.where(self.n_sampled<self.max_n_sample)[0]
        # Fit a Gaussian Process model to the sampled data
        self._fit_gp_model(n_restarts)

    
    def get_nrmse_and_dist(self, gp_gt, **kwargs):
        locs_np = self.all_locs.detach().numpy()
        meps_pred = self.gp_model.predict(self.all_locs)[0].reshape(-1).detach().numpy()
        meps_gt = gp_gt.predict(self.all_locs)[0].reshape(-1).detach().numpy()
        nmse = np.linalg.norm(meps_gt-meps_pred)**2/np.linalg.norm(meps_gt)**2

        cog_model = u.cog(locs_np, meps_pred, **kwargs)
        cog_gt = u.cog(locs_np, meps_gt, **kwargs)
        
        dist_loc = np.sqrt(((cog_model[:2]-cog_gt[:2])**2).sum())

        if self.all_locs.shape[-1] == 4:
            cog_ang = np.arctan2(cog_model[2],cog_model[3])*180/np.pi
            cog_ang_gt = np.arctan2(cog_gt[2],cog_gt[3])*180/np.pi
            dist_ang = abs(cog_ang-cog_ang_gt)

            return nmse, dist_loc, dist_ang
        
        if self.all_locs.shape[-1] == 3:
            dist_ang = abs(cog_model[2]-cog_gt[2])

            return nmse, dist_loc, dist_ang
        else:
            return nmse, dist_loc

    def _fit_gp_model(self, n_restarts=1):
        X = self.all_locs[self.sampled_loc_idcs]
        y = self.sampled_returns.reshape(-1,1)
        self.gp_model = CustomSquareWarpedGP(X, y)
        self.gp_model.fit(n_restarts=n_restarts)


    def _acquisition_selection(self):
        # Select the next state using the specified acquisition function
        if self.acquisition_function == 'UCB':
            next_state = self._ucb_selection()
        elif self.acquisition_function == 'EI':
            next_state = self._ei_selection()
        elif self.acquisition_function == 'PI':
            next_state = self._pi_selection()
        elif self.acquisition_function == 'MVE':
            next_state = self._mve()
        elif self.acquisition_function == 'TS':
            next_state = self._thompson_sampling_selection()
        elif self.acquisition_function == 'random':
            next_state = self._random_sampling()
        elif self.acquisition_function == 'KG':
            next_state = self._knowledge_gradient()
        else:
            raise ValueError("Invalid acquisition function: %s" % self.acquisition_function)

        return next_state
    
    def _random_sampling(self):
        idx = torch.randint(len(self.available_loc_idcs),(1,))[0]
        next_state_index = self.available_loc_idcs[idx]
        return next_state_index
    
    def _knowledge_gradient(self):        


        max_pmean = torch.max(self.gp_model.posterior(self.all_locs).mean)

        qKG = qKnowledgeGradient(self.gp_model, num_fantasies=NUM_FANTASIES)

        qKG_proper = qKnowledgeGradient(
            self.gp_model,
            num_fantasies=NUM_FANTASIES,
            sampler=qKG.sampler,
            current_value=max_pmean,
        )

        candidates_proper, _ = optimize_acqf(
            acq_function=qKG_proper,
            bounds=BOUNDS,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )
        dists = torch.norm(self.all_locs[self.available_loc_idcs]-candidates_proper, dim=-1)
        next_state_index = self.available_loc_idcs[torch.argmin(dists)]
        return next_state_index
    
    def _mve(self):

        MES = qMaxValueEntropy(model=self.gp_model, candidate_set=self.all_locs)
        mes = MES(self.all_locs[self.available_loc_idcs].unsqueeze(1))
        next_state_index = self.available_loc_idcs[torch.argmax(mes)]
        return next_state_index


    def _ucb_selection(self):

        Ucb = analytic.UpperConfidenceBound(model=self.gp_model, beta=10+np.e**(-len(self.sampled_loc_idcs)/10+2))

        ucbs = Ucb(self.all_locs[self.available_loc_idcs].unsqueeze(1))
        next_state_index = self.available_loc_idcs[np.random.choice(np.argwhere(ucbs==ucbs.max())[0])]
        return next_state_index
    
    def _ei_selection(self):
        max_pmean = torch.max(self.gp_model.posterior(self.all_locs).mean)

        LogEI = analytic.LogExpectedImprovement(model=self.gp_model, best_f=max_pmean)

        log_eis = LogEI(self.all_locs[self.available_loc_idcs].unsqueeze(1))
        next_state_index = self.available_loc_idcs[torch.argmax(log_eis)]
        return next_state_index

    def _pi_selection(self):
        max_pmean = torch.max(self.gp_model.posterior(self.all_locs).mean)

        Lpi = analytic.LogProbabilityOfImprovement(model=self.gp_model, best_f=max_pmean)

        log_pis = Lpi(self.all_locs[self.available_loc_idcs].unsqueeze(1))
        next_state_index = self.available_loc_idcs[torch.argmax(log_pis)]
        return next_state_index

    def _thompson_sampling_selection(self):
        posterior = self.gp_model.posterior(self.all_locs[self.available_loc_idcs].unsqueeze(1))
        thompson_sample = posterior.sample()[0,:,0]
        next_state_index = torch.argmax(thompson_sample)
        return self.available_loc_idcs[next_state_index]
