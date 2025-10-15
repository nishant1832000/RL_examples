#!/usr/bin/env python3
import gymnasium as gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter
from gymnasium.wrappers import RecordVideo,TimeLimit


TimeLimit.max_episode_steps = 2000


import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 228
BATCH_SIZE = 18
PERCENTILE = 75


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs,info = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]

        # print("act_probs :",act_probs)
        # action = np.argmax(act_probs)
        action = np.random.choice(len(act_probs), p=act_probs)

        # print("action :",action)


        next_obs, reward, terminated, truncated, info = env.step(action)

        # print("reward :",reward)
        episode_reward += reward


        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if terminated or truncated:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs,info = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)

    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1",render_mode="rgb_array")
    
    env = RecordVideo(
            env,
            video_folder="mon",          # Directory to save videos
            episode_trigger=lambda e:e % 50 == 0,  # Record every episode (or customize)
            name_prefix="episode"        # Optional: prefix for filenames
        )
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    # print("n_actions :",n_actions)


    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.007)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 499:
            print("Solved!")
            break
    writer.close()
