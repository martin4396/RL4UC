import warnings
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import time

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# * define net
net = pn.case24_ieee_rts()
num_unit = len(net.poly_cost)

# * define hyperparameters
changeTimesLimit = 4
stillLimit = 2
RESERVE = .05
RAMPING_LIMIT = .4
RESERVE_PENALTY = 50000
FOWARD_REWARD = 200000


def get_data(name, power=False):
    # can directly get the answer
    if power == False:
        if name in net['poly_cost'].columns:
            return net['poly_cost'][name]
        else:
            ans = []
            for i in range(len(net['poly_cost'])):
                try:
                    point = net['poly_cost'].iloc[i]
                    et, element = point['et'], point['element']
                    ans.append(net[et][name][element])
                except:
                    ans.append(0)
                    # r1aise warnings.warn('Something went wrong when retrieving %s data. ' % (name) +
                    #                 'Check poly_cost %d for more details' % (i))
            return torch.tensor(ans, device=device)
    else:
        ans = []
        for i in range(len(net['poly_cost'])):
            try:
                point = net['poly_cost'].iloc[i]
                et, element = point['et'], point['element']
                ans.append(net['res_'+et]['p_mw'][element])
            except:
                ans.append(0)
        return torch.tensor(ans, device=device)


def save_data(name, data):
    for i in range(len(net['poly_cost'])):
        try:
            point = net.poly_cost.iloc[i]
            et, element = point.et, int(point.element)
            net[et][name][element] = data[i].item()
        except:
            raise Exception('Something went wrong when saving %s data. ' % (name) +
                            'Check poly_cost %d for more details' % (i))


# * define hyperparameters
power_load_ratio = .9
load_multiplier = get_data('max_p_mw').sum().item()/power_load_ratio/net.load['p_mw'].sum()
total_load = get_data('max_p_mw').sum().item()/power_load_ratio


RAMPING = get_data('max_p_mw') * RAMPING_LIMIT
INIT_MAX = get_data('max_p_mw')
INIT_MIN = get_data('min_p_mw')


# * define dataset
class UCDataset(Dataset):
    def __init__(self, file_name, seed=None, num_ft_load=6, name=None):
        super(UCDataset, self).__init__()
        if name is None:
            raise Exception('Please specify a name for the dataset')

        self.num_ft_load = num_ft_load
        self.name = name
        # reading load curve data file
        data = pd.read_csv(file_name, index_col=0)
        data = data.div(data.max().max())
        # exclude the unsucessful case by matlab
        if file_name == 'imitate_load_power.csv':
            available = sorted(
                list(map(lambda x: x.split('.')[0], os.listdir('data/onoff/'))))
            data = data.iloc[available]
        self.orig_index = data.index
        self.data = data.reset_index(drop=True)
        self.idx = self.data.index

        # num_samples represents the number of the total training data
        self.num_samples = num_samples = len(self.data)

        # set the random seed
        if seed is None:
            seed = np.random.randint(1, 1234)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cross_section_shape = (num_unit,)

        # * generate static features
        self.num_static = 5
        # ! require that all data in the sequence of poly_cost
        # define the power cost of each unit
        cost2 = get_data('cp2_eur_per_mw2')
        cost1 = get_data('cp1_eur_per_mw')
        cost0 = get_data('cp0_eur')
        # define the minimum on/off time limit before changing states of each unit
        minTimeLimit = torch.full(cross_section_shape, 2, dtype=torch.int)
        # define reserve rate of the unit
        reserveRate = torch.full(cross_section_shape, 1+RESERVE)
        # compose the above static data
        static = torch.tensor(
            np.stack([cost2, cost1, cost0, minTimeLimit, reserveRate])
        ).expand(num_samples, -1, -1)
        self.static = static

        # * generate dynamic features
        self.num_dynamic = 3 + num_ft_load
        # define the states of the unit
        state = torch.ones(cross_section_shape)
        # define the realized on/off time
        time = torch.full(cross_section_shape, 12, dtype=torch.int)
        # define the last period power
        power = get_data('p_mw').cpu()
        # compose the above dynamic data
        dynamic = torch.tensor(
            np.stack([state, time, power])
        ).expand(num_samples, -1, -1)
        # * define the load curve
        load = torch.tensor([self.data.iloc[:, k] for k in range(
            num_ft_load)]).expand(33, -1, -1).permute(2, 1, 0)
        dynamic = torch.cat([dynamic, load], dim=1)
        self.dynamic = dynamic

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.static[idx], self.dynamic[idx], self.idx[idx]

    def update_mask(self, dynamic, unit_state, nOchg):
        mask = torch.ones_like(unit_state)
        zero = torch.zeros_like(mask)
        # changing times limit
        mask = torch.where(nOchg >= changeTimesLimit, zero, mask)
        # still limit
        stillTime = torch.abs(dynamic[:, 1, :])
        mask = torch.where(stillTime <= stillLimit, zero, mask)
        return mask

    def update_dynamic(self, dynamic, idx, unit_state, power, moment, nOchg):
        # dynamic [256,9,33]
        ## state, time, power, future_load
        num_ft_load = self.num_ft_load
        stateChg = torch.ne(unit_state, dynamic[:, 0, :])
        nOchg = nOchg + stateChg
        # state
        dynamic[:, 0, :] = unit_state
        # time
        if moment == 0:
            ones = torch.ones_like(dynamic[:, 1, :])
            dynamic[:, 1, :] = torch.where(unit_state == 1, ones, -ones)
        else:
            unchanged = dynamic[:, 1, :] + torch.sign(unit_state-0.5)
            changed = torch.sign(unit_state-0.5).double()
            dynamic[:, 1, :] = torch.where(stateChg, changed, unchanged)
        # power
        dynamic[:, 2, :] = power
        # future_load
        load_range = torch.clamp(torch.arange(
            moment+1, moment+num_ft_load+1), 0, 23)
        new_load = torch.tensor(self.data.iloc[idx, load_range].values).expand(
            33, -1, -1).permute(1, 2, 0)
        dynamic[:, 3:, :] = new_load
        return dynamic, nOchg


def reward(unit_state, last_power, load, last_success):
    # ramping limit and calculate penalty for reserve limit
    cost, true_cost, power, success = [], [], [], []
    for case_idx in range(unit_state.shape[0]):
        if last_success[case_idx]==False:
            cost.append(0)
            true_cost.append(500000.)
            power.append(torch.zeros((33,),device=device))
            success.append(False)
            continue

        max_power = last_power[case_idx] + RAMPING
        min_power = last_power[case_idx] - RAMPING
        max_power = np.array([x.item() if x < y else y.item()
                              for x, y in zip(max_power, INIT_MAX)])
        min_power = np.array([y.item() if x < y else x.item()
                              for x, y in zip(min_power, INIT_MIN)])
        penalty = 0

        cond1=max_power[unit_state[case_idx].bool().cpu()].sum() < load[case_idx]*(1+RESERVE)*total_load
        cond2=min_power[unit_state[case_idx].bool().cpu()].sum() > load[case_idx]*(1-RESERVE)*total_load
        if cond1:
            penalty += RESERVE_PENALTY
        if cond2:
            penalty += RESERVE_PENALTY
        save_data('max_p_mw', max_power)
        save_data('min_p_mw', min_power)

        net.load['scaling'] = load[case_idx].item() * load_multiplier
        save_data('in_service',unit_state[case_idx].bool())


        try:
            pp.rundcopp(net, delta=1e-16)
            cost.append(-net.res_cost - penalty + FOWARD_REWARD)
            true_cost.append(net.res_cost)
            power.append(get_data('p_mw', power=True))
            success.append(True)
        except:
            cost.append(0)
            true_cost.append(500000.)
            # * how to deal with unsuccessful scenario
            power.append(torch.zeros((33,), device=device))
            success.append(False)

    return torch.tensor(cost,dtype=float,device=device), torch.stack(power).to(device), torch.tensor(success,device=device), torch.tensor(true_cost,device=device)
