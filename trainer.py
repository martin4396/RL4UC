import os
import time
import random
import argparse
import datetime
from typing_extensions import final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pprint import pprint
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, RandomSampler

from model import Encoder, RL4UC

# -------------------------------------------------------------------------
# global parameters define
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
now_time = time.strftime("%m%d_%H%M%S", time.localtime())
log_name = 'power_log/'+now_time+'.txt'
plt.style.use('ggplot')
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# utils function
def log_output(words: object = '', others=None):
    # * write text into log file
    # write the words on the screen and log_file
    print(words)
    if others is not None:
        print(others)
    with open(log_name, 'a') as f:
        f.write(words+'\n')
        if others is not None:
            f.write(others+'\n')


def dataset_split(dataset, ratio):
    # * split dataset into 2 parts, and part1 has ratio% of the data
    num_data = len(dataset)
    num_part1 = round(num_data*ratio)
    num_part2 = num_data-num_part1
    part1, part2 = random_split(dataset, [num_part1, num_part2])
    return part1, part2


def get_matlab_ans(idxs):
    from utils import matlab2pp
    failures1, failures2, onoff, power = [], [], [], []
    artificial_state = np.full((33, 24), 0.5)
    artificial_power = np.full((33, 24), 0.)
    for idx in idxs:
        file_name = 'data/onoff/'+str(idx.item())+'.csv'
        if not os.path.exists(file_name):
            failures1.append(idx)
            onoff.append(artificial_state)
        else:
            unit_state = matlab2pp(pd.read_csv(file_name, header=None))
            onoff.append(unit_state)
        file_name = 'data/power/'+str(idx.item())+'.csv'
        if not os.path.exists(file_name):
            failures2.append(idx)
            power.append(artificial_power)
        else:
            unit_power = matlab2pp(pd.read_csv(file_name, header=None))
            power.append(unit_power)
    onoff = torch.tensor(
        np.stack(onoff), dtype=torch.float32, device=device)
    power = torch.tensor(
        np.stack(power), dtype=torch.float32, device=device)
    return onoff, power, failures1, failures2
def get_load_curve(idxs, dataset,test=False):
    # find the origin dataset
    if test: 
        return dataset.data.loc[idxs]
    else: 
        return dataset.dataset.data.loc[idxs]
def update_dynamic(dynamic, matlab_ans, matlab_power, moment, load_curve):
    # from UC import reward
    # dynamic [256,9,33]  matlab_ans [256,33,24]
    ## state, time, power, future_load
    num_ft_load = dynamic.shape[1] - 3
    # state
    dynamic[:, 0, :] = matlab_ans[:, :, moment]
    # time
    if moment == 0:
        ones = torch.ones_like(dynamic[:, 1, :])
        dynamic[:, 1, :] = torch.where(
            matlab_ans[:, :, moment] == 1, ones, -ones)
    else:
        stateChg = torch.ne(
            matlab_ans[:, :, moment], matlab_ans[:, :, moment-1])
        unchanged = (dynamic[:, 1, :] +
                     torch.sign(matlab_ans[:, :, moment]-0.5)).float()
        changed = (torch.sign(matlab_ans[:, :, moment]-0.5)).float()
        dynamic[:, 1, :] = torch.where(stateChg, changed, unchanged)
    # power
    dynamic[:, 2, :] = matlab_power[:, :, moment]
    # future_load
    load_range = torch.clamp(torch.arange(
        moment+1, moment+num_ft_load+1), 0, 23)
    new_load = torch.tensor(load_curve.iloc[:, load_range].values).expand(
        33, -1, -1).permute(1, 2, 0)
    dynamic[:, 3:, :] = new_load
    return dynamic
# ------------------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(Critic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # define the encoder and decoder models
        self.fc1 = nn.Conv1d(2*hidden_size, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, tour_dynamic):
        # use the probability of unit
        ans = []
        static_hidden = self.static_encoder(static)
        for i in range(tour_dynamic.shape[1]):
            dynamic = tour_dynamic[:, i, :, :]
            dynamic_hidden = self.dynamic_encoder(dynamic)

            hidden = torch.cat((static_hidden, dynamic_hidden), 1)

            output = F.relu(self.fc1(hidden))
            output = F.relu(self.fc2(output))
            output = self.fc3(output).sum(dim=2)
            ans.append(output)
        return torch.cat(ans, axis=1).to(device)


def validate(dataloader, actor, mode='train'):
    if mode == 'train':
        actor.eval()
        rewards = []
        for batch_idx, batch in enumerate(dataloader):
            static, dynamic, idx = batch
            static, dynamic = static.to(device), dynamic.to(device)
            with torch.no_grad():
                _, _, _, success, _, true_cost = actor(static, dynamic, idx, batch_idx)
                # episode_logp, episode_cost, episode_power, episode_success, tour_dynamic
            rewards.append(torch.mean((torch.sum(true_cost)/torch.sum(success)).detach()).item())
        actor.train()
        return np.mean(rewards)
    elif mode == 'test':
        actor.eval()
        rewards, powers, successes, true_costs = [], [], [], []
        for batch_idx, batch in enumerate(dataloader):
            static, dynamic, idx = batch
            static, dynamic = static.to(device), dynamic.to(device)
            with torch.no_grad():
                _, reward, power, success, _,true_cost = actor(
                    static, dynamic, idx, batch_idx, mode='test')
                # logp, reward, power, success, tour_dynamic, true_cost
            rewards.append(reward.mean().item())
            powers.append(power.mean().item())
            successes.append(success.float().mean().item())
            true_costs.append(true_cost.mean().item())
        return rewards, powers, successes, true_costs
    else:
        raise Exception('Unknown validate mode, expect train or test')


def imitate(actor, imitate_train_dataset, imitate_test_dataset, batch_size, imitate_lr, **kwargs):


    # * imitate begin
    epoch_loss, epoch_test_loss, epoch_time = [], [], []

    actor_optim = optim.Adam(actor.Actor.parameters(), lr=imitate_lr)
    criterion = nn.MSELoss()
    train_data, test_data = imitate_train_dataset, imitate_test_dataset
    train_loader = DataLoader(
        train_data, batch_size, shuffle=False, sampler=RandomSampler(train_data, replacement=True, num_samples=len(train_data)), num_workers=8)
    test_loader = DataLoader(
        test_data, batch_size, False, sampler=RandomSampler(test_data, replacement=True, num_samples=len(test_data)), num_workers=8)

    log_output(
        'imitate learning begin\n----------------------------------------------------')

    PATH = 'state_dict/imitate_actor_parameter_0521_001633.pt'
    actor.load_state_dict(torch.load(PATH))

    num_epoch = 10
    for epoch in range(num_epoch):
        actor.train()
        actor.Actor.to(device)
        times, losses, test_losses = [], [], []
        log_output('    epoch %d begin' % (epoch+1))

        # # imitate train the matlab answer
        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()
            static, dynamic, idx = batch
            static, dynamic = static.to(device), dynamic.to(device)
            running_loss = 0
            matlab_ans, matlab_power, _, _ = get_matlab_ans(idx)  # [256,33,24]
            train_load_curve = get_load_curve(idx, train_data)
            for moment in range(24):
                # calculate unit_state and backward
                unit_state = actor.Actor(static, dynamic)  # unit state[256,33]
                labels = matlab_ans[:, :, moment]
                loss = criterion(unit_state, labels)
                actor_optim.zero_grad()
                loss.backward()
                actor_optim.step()
                running_loss += loss.item()
                # update dynamic state
                dynamic = update_dynamic(
                    dynamic, matlab_ans, matlab_power, moment, train_load_curve)
            losses.append(running_loss)
            times.append(time.time()-start_time)
            if (batch_idx % 10 == 0) or (batch_idx == len(train_loader)-1):
                log_output('        batch %d loss = %.6f, time = %.4fs'
                           % (batch_idx+1, losses[-1], times[-1]))
        epoch_loss.append(np.mean(losses))
        epoch_time.append(np.sum(times))

        # imitate test the matlab answer
        actor.eval()
        for batch_idx, batch in enumerate(test_loader):
            static, dynamic, idx = batch
            static, dynamic = static.to(device), dynamic.to(device)
            running_loss = 0
            matlab_ans, matlab_power, _, _ = get_matlab_ans(idx)  # [256,33,24]
            test_load_curve = get_load_curve(idx, test_data)
            for moment in range(24):
                # calculate unit_state and backward
                unit_state = actor.Actor(static, dynamic)  # [256,33]
                labels = matlab_ans[:, :, moment]
                loss = criterion(unit_state, labels)
                running_loss += loss.item()
                # update dynamic state
                dynamic = update_dynamic(
                    dynamic, matlab_ans, matlab_power, moment, test_load_curve)
            test_losses.append(running_loss)

        epoch_test_loss.append(np.mean(test_losses))

        # log_output('    epoch %d ends with avg train loss = %.6f, avg test loss = %.6f, total time = %.4fs'
        #            % (epoch+1, epoch_loss[-1], epoch_test_loss[-1], epoch_time[-1]))
        log_output('    epoch %d ends with avg avg test loss = %.6f'
                   % (epoch+1,  epoch_test_loss[-1]))

    # * saving results as csv & print log
    result = pd.DataFrame({'imitate_train_loss': epoch_loss, 'imitate_test_loss': epoch_test_loss,
                           'epoch_time': epoch_time}, index=['epoch_%d' % (k+1) for k in range(num_epoch)])
    result.to_csv('result/imitate/'+now_time+'lr%f' % (imitate_lr)+'.csv')

    log_output('Result: avg train loss = %.6f, avg test loss = %.6f, total time = %.4fs'
               % (np.mean(epoch_loss), np.mean(epoch_test_loss), np.sum(epoch_time)))
    log_output(
        '----------------------------------------------------\nimitate learning end')

    # * drawing pictures
    figure_name = 'figure/'+now_time+'lr%f' % (imitate_lr)+'.png'
    plt.figure(figsize=(9, 15))
    plt.subplot(311)
    plt.plot(epoch_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss on training set per epoch,lr={}'.format(imitate_lr))
    plt.subplot(312)
    plt.plot(epoch_test_loss)
    plt.xlabel('epoch')
    plt.ylabel('test loss')
    plt.title('loss on test set per epoch')
    plt.subplot(313)
    plt.plot(epoch_time)
    plt.xlabel('epoch')
    plt.ylabel('time(s)')
    plt.title('time per epoch')
    plt.savefig(figure_name)

    # * save the model
    torch.save(actor.state_dict(),
               'state_dict/imitate_actor_parameter_{}.pt'.format(now_time))
    log_output('Actor is saved at {}.'.format(now_time))


def train(actor, critic, actor_lr, critic_lr, reinforce_dataset, true_dataset, batch_size,
          max_grad_norm, **kwargs):
    def extract_loss(data, idx):
        ans = torch.zeros(data.shape[0])
        for i in range(data.shape[0]):
            ans[i] = torch.sum(data[i, :idx[i]+1])
        return torch.mean(ans)
    def compute_returns(rewards,success,gamma=1.0):
        R = torch.zeros(rewards.shape[0],device=device)
        returns=[]
        for step in reversed(range(rewards.shape[1])):
            R = rewards[:,step]+gamma*R*success[:,step]
            returns.insert(0,R.view(-1,1))
        returns=torch.cat(returns,dim=1)
        return returns.to(device)

    # * load actor from the imitate process
    PATH = 'state_dict/imitate_actor_parameter_0521_001633.pt'
    
    actor.load_state_dict(torch.load(PATH))

    # for p in actor.Actor.model.parameters():
    #     if p not in actor.Actor.model.fc.parameters(): p.requires_grad=False

    save_dir = os.path.join(
        'save', 'alr={}_clr={}'.format(actor_lr, critic_lr))
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # * optim & dataloader preparation
    actor_optim = optim.Adam(filter(lambda p: p.requires_grad, actor.parameters()), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(
        reinforce_dataset, batch_size, shuffle=False, sampler=RandomSampler(reinforce_dataset, replacement=True, num_samples=len(reinforce_dataset)), num_workers=4)
    valid_loader = DataLoader(
        true_dataset, batch_size, shuffle=False, sampler=RandomSampler(true_dataset, replacement=True, num_samples=len(true_dataset)), num_workers=4)

    best_epoch = None
    best_reward = 0
    # reward代表加上奖励与惩罚的值，cost代表opf计算的机组成本，walk代表平均步数
    epoch_loss,epoch_reward, epoch_cost, epoch_time, epoch_valid, epoch_walk = [], [], [], [], [], []
    epoch_max_walk = []

    log_output(
        'reinforce learning begin\n----------------------------------------------------')
    # * training
    num_epoch = 30
    for epoch in range(num_epoch):
        actor.train()
        critic.train()

        # longest代表最长步数
        times, losses, rewards, walk, cost, longest = [], [], [], [], [], []

        start = epoch_start = time.time()

        log_output('    epoch %d begin' % (epoch+1))

        for batch_idx, batch in enumerate(train_loader):
            static, dynamic, idx = batch
            static, dynamic = static.to(device), dynamic.to(device)

            # full forward pass through the dataset
            logp, reward, _, success, tour_dynamic, true_cost = actor(
                static, dynamic, idx, batch_idx)
            assert (reward<0).sum() == 0
            # * reward is implemented in the actor

            critic_est = critic(static, tour_dynamic)

            # * calculate returns of each episode

            returns=compute_returns(reward,success).detach()

            #* backward start
            episode_length = torch.sum(success, axis=1)
            advantage = returns-critic_est
            tour_logp = logp.sum(axis=2)

            actor_loss = extract_loss(
                -advantage.detach()*tour_logp, episode_length)
            critic_loss = extract_loss(advantage**2, episode_length)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()
            #* backward end

            # critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(critic_loss.detach()).item())
            walk.append((torch.sum(success)/float(batch_size)).detach().item())
            longest.append(success.sum(axis=1).max().detach().item())
            cost.append(torch.mean(true_cost.detach()).item())

            report_num = 25
            if (batch_idx+1) % report_num == 0:
                end = time.time()
                times.append(end-start)
                start = end

                mean_loss = np.mean(losses[-report_num:])
                mean_reward = np.mean(rewards[-report_num:])
                mean_walk = np.mean(walk[-report_num:])
                mean_cost = np.mean(cost[-report_num:])
                max_longest = np.max(longest[-report_num:])

                log_output('        batch %d reward = %2.3f, true_cost = %2.3f, loss = %2.4f, mean walk = %2.4f, max walk = %d, time = %2.4fs' %
                           (batch_idx+1,  mean_reward, mean_cost, mean_loss, mean_walk,max_longest, times[-1]/report_num))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        mean_walk = np.mean(walk)
        mean_cost = np.mean(cost)
        max_longest = np.max(longest)

        # * save the weights
        epoch_dir = os.path.join(checkpoint_dir, 'actor.pt')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # * validation
        # valid_dir = os.path.join(save_dir, '%s' % epoch)
        mean_valid,_,_,_ = validate(valid_loader, actor, mode='test')
        mean_valid=np.mean(mean_valid)
        if mean_valid > best_reward:
            best_reward = mean_valid
            best_epoch = epoch
            valid_path = os.path.join(
                'state_dict_new3', 'reinforce_alr={}_clr={}_{}'.format(actor_lr, critic_lr,now_time))
            if not os.path.exists(valid_path):
                os.makedirs(valid_path)

            save_path = os.path.join(valid_path, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(valid_path, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        log_output('    Mean epoch loss = %2.4f, reward = %2.4f, cost = %2.4f, valid_reward = %2.4f, mean walk = %2.4f, max walk = %d,  time = %2.4fs' %
                   (mean_loss, mean_reward, mean_cost, mean_valid, mean_walk,max_longest, time.time()-epoch_start))

        # * record the result
        epoch_loss.append(mean_loss)
        epoch_reward.append(mean_reward)
        epoch_cost.append(mean_cost)
        epoch_valid.append(mean_valid)
        epoch_walk.append(mean_walk)
        epoch_time.append(time.time()-epoch_start)
        epoch_max_walk.append(max_longest)
            # * save the result
        result = pd.DataFrame({'epoch_loss': epoch_loss,'epoch_reward':epoch_reward, 'epoch_cost': epoch_cost,
                                'epoch_valid': epoch_valid, 'epoch_walk':epoch_walk, 'epoch_max_walk':epoch_max_walk,'epoch_time': epoch_time},
                            index=['%d' % (k+1) for k in range(len(epoch_loss))])
        result.to_csv('result/reinforce/'+now_time+'alr%fclr%f' %
                    (actor_lr, critic_lr)+'.csv')

        torch.cuda.empty_cache()

    log_output(
        '-----------------------------------------------------------------------')
    log_output('Best validation epoch is {}'.format(best_epoch))
    # * save the result
    result = pd.DataFrame({'epoch_loss': epoch_loss,'epoch_reward':epoch_reward, 'epoch_cost': epoch_cost,
                             'epoch_valid': epoch_valid, 'epoch_walk':epoch_walk, 'epoch_time': epoch_time},
                          index=['%d' % (k+1) for k in range(num_epoch)])
    result.to_csv('result/reinforce/'+now_time+'alr%fclr%f' %
                  (actor_lr, critic_lr)+'.csv')
    # * plot the figures
    figure_name = 'train_figure/'+now_time + \
        'alr%fclr%f' % (actor_lr, critic_lr)+'.png'
    plt.figure(figsize=(24, 18))
    for idx, column in enumerate(result.columns):
        plt.subplot(3, 2, idx+1)
        result[column].plot()
        plt.title(str(column))
        plt.xlabel('epoch')
    plt.savefig(figure_name)

@torch.no_grad()
def test(actor, test, true_dataset, **kwargs):
    if test == 'imitate':
        PATH = 'state_dict/imitate_actor_parameter_0521_001633.pt'
        actor.load_state_dict(torch.load(PATH))
        test_loader= DataLoader(true_dataset,batch_size=16)
        criterion = nn.MSELoss()
        test_losses=[]
        episode_label,episode_states=[],[]
        for batch_idx, batch in enumerate(test_loader):
            static, dynamic, idx = batch
            static, dynamic = static.to(device), dynamic.to(device)
            running_loss = 0
            matlab_ans, matlab_power, _, _ = get_matlab_ans(idx)  # [256,33,24]
            test_load_curve = get_load_curve(idx, true_dataset,test=True)
            unit_states=torch.zeros_like(matlab_ans)
            for moment in range(24):
                # calculate unit_state and backward
                unit_state = actor.Actor(static, dynamic)  # [256,33]
                label = matlab_ans[:, :, moment]
                loss = criterion(unit_state, label)
                running_loss += loss.item()
                # update dynamic state
                dynamic = update_dynamic(
                    dynamic, matlab_ans, matlab_power, moment, test_load_curve)
                unit_states[:,:,moment]=unit_state
            test_losses.append(running_loss)
            for case_idx in idx:
                case_idx=case_idx.item()
                save_path=os.path.join('imitate_test',str(case_idx))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path1=os.path.join('imitate_test',str(case_idx),'imitate.csv')
                save_path2=os.path.join('imitate_test',str(case_idx),'result.csv')
                save_path3=os.path.join('imitate_test',str(case_idx),'difference.csv')
                
                imitate=pd.DataFrame(unit_states[case_idx%16,:, :].cpu().numpy())
                result=pd.DataFrame(matlab_ans[case_idx%16,:,:].cpu().numpy())
                result=result.astype(int)
                difference=imitate-result
                imitate.to_csv(save_path1)
                result.to_csv(save_path2)
                difference.to_csv(save_path3)
        return 
    elif test=='reinforce':
        # TODO reinforce result
        PATH = 'state_dict_new0.3/reinforce_alr=0.0005_clr=0.0005_0610_144515'
        PATH = PATH+'/actor.pt'
        actor.load_state_dict(torch.load(PATH))
        test_loader= DataLoader(true_dataset,batch_size=16)

        actor.eval()
        rewards, powers, successes, true_costs = [], [], [], []
        walk,times=[],[]
        for batch_idx, batch in enumerate(test_loader):
            static, dynamic, idx = batch
            static, dynamic = static.to(device), dynamic.to(device)
            start=time.time()
            with torch.no_grad():
                _, reward, power, success, _,true_cost = actor(
                    static, dynamic, idx, batch_idx, mode='test')
                # logp, reward, power, success, tour_dynamic, true_cost
                
            times.append(time.time()-start)
            walk.append(reward.shape[1])
            rewards.append(reward)
            powers.append(power)
            successes.append(success)
            true_costs.append(true_cost)

        return

def train_init(args):
    import UC
    from UC import UCDataset

    # form imitate dataset and reinforce dataset
    reinforce_dataset = UCDataset('reinforce_load.csv', name='reinforce')
    imitate_dataset = UCDataset('imitate_load_power.csv', name='imitate')
    imitate_train_dataset, imitate_test_dataset = dataset_split(
        imitate_dataset, 0.9)
    # imitate period test
    # period_test_dataset = UCDataset('imitate_test.csv', name='imitate_test')
    # form test dataset and valid dataset
    true_dataset = UCDataset('load.csv', name='trueData')

    actor = RL4UC(
        reinforce_dataset.num_static,
        reinforce_dataset.num_dynamic,
        args.hidden_size,
        args.batch_size,
        reinforce_dataset.update_mask,
        reinforce_dataset.update_dynamic,
        UC.reward,
        alpha=args.alpha
    ).to(device)

    critic = Critic(
        reinforce_dataset.num_static,
        reinforce_dataset.num_dynamic,
        args.hidden_size
    ).to(device)

    kwargs = vars(args)
    kwargs['imitate_train_dataset'] = imitate_train_dataset
    kwargs['imitate_test_dataset'] = imitate_test_dataset
    kwargs['reinforce_dataset'] = reinforce_dataset
    kwargs['true_dataset'] = true_dataset
    # kwargs['period_test_dataset'] = period_test_dataset
    # kwargs['test_dataset'] = test_dataset
    # kwargs['valid_dataset'] = valid_dataset

    # if args.checkpoint:
    #     PATH = 'state_dict/imitate_actor_parameter_0521_001633.pt'
    #     actor.load_state_dict(torch.load(PATH))


        # path = os.path.join(args.checkpoint, 'actor.pt')
        # actor.load_state_dict(torch.load(path, device))

        # path = os.path.join(args.checkpoint, 'critic.pt')
        # critic.load_state_dict(torch.load(path, device))

    if not args.test:
        # imitate(actor, **kwargs)
        train(actor, critic, **kwargs)
    else:
        test(actor, **kwargs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Solving Unit Commitment problem')
    parser.add_argument('--seed', default=12345, type=int)
    # parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--imitate_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    # parser.add_argument('--test_size', default=8, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--ft_load', default=6, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--log', default=True, type=bool)
    args = parser.parse_args()

    # set GPUs
    localtime1 = time.asctime(time.localtime(time.time()))
    log_output('{}'.format(localtime1))
    torch.cuda.set_device(args.gpu)
    log_output('gpu is {}'.format(args.gpu))
    log_output('{}'.format(args))

    try:
        train_init(args)
    except Exception as e:
        log_output('Exit due to exception {}'.format(e))
    finally:
        localtime2 = time.asctime(time.localtime(time.time()))
        log_output('{}'.format(localtime1))
        log_output('{}'.format(localtime2))
