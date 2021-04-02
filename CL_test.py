import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import kornia.augmentation as aug
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
import cv2 as cv

BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MOMENTUM_UPDATE_ITER = 10
MEMORY_CAPACITY = 1000
OBSERV_MEMORY_CAPACITY = 50
random_shift = nn.Sequential(aug.RandomCrop((80, 80)), nn.ReplicationPad2d(4), aug.RandomCrop((84, 84)))
img = cv.imread('/home/sam/Pictures/atari_background.jpg')
transf = transforms.ToTensor()
img_tensor = transf(img)
R  = img_tensor[0]
G  = img_tensor[1]
B  = img_tensor[2]
img_tensor[0]=0.299*R+0.587*G+0.114*B
ATARI_CANVAS = np.ceil(img_tensor[0][200:300, 500:600]*255)

class Box2DEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.target_x = 0
        self.L = 50
        self.l = 5
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(np.array([0,0]), np.array([2*self.L-1, 2*self.L-1]))
        self.observ = None
        self.latent_state = None

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        x, y = self.latent_state
        if action == 0:
            x = x - 1
            y = y
        if action == 1:
            x = x + 1
            y = y
        self.latent_state = np.array([x, y])
        canvas = torch.ones(2*self.L, 2*self.L)*150
        #canvas = ATARI_CANVAS
        x, y = int(x), int(y)
        canvas[x-self.l:x+self.l+1, y-self.l:y+self.l+1] = 255
        self.observ = canvas
        done = (x <= self.l) or ( x+self.l >= 2 * self.L -1)
        done = bool(done)
        if not done:
            reward = -0.1
        else:
            if x <= self.l:
                reward = 10
            else:
                reward = -10
        return self.observ, reward, done, {}

    def reset(self):
        #self.latent_state = np.ceil(np.ceil(np.ceil(np.random.rand(2) * (2 * self.L-self.l))+self.l))
        self.latent_state = (50,50)
        canvas = torch.ones(2*self.L, 2*self.L)*150
        #canvas = ATARI_CANVAS
        x, y = self.latent_state
        x, y = int(x), int(y)
        canvas[x-self.l:x+self.l+1, y-self.l:y+self.l+1] = 255
        self.observ = canvas
        return self.observ

class MovingBox2DEnv(Box2DEnv):
    def __init__(self):
        super(MovingBox2DEnv, self).__init__()
        self.observ = None
        self.latent_state = None
        self.movebox_latent_state = None

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        x, y = self.latent_state
        if action == 0:
            x = x - 1
            y = y
        if action == 1:
            x = x + 1
            y = y
        self.latent_state = np.array([x, y])
        #canvas = torch.ones(2*self.L, 2*self.L)*150
        canvas = ATARI_CANVAS
        x, y = int(x), int(y)
        canvas[x-self.l:x+self.l+1, y-self.l:y+self.l+1] = 255
        x_movebox, y_movebox = self.movebox_latent_state
        x_movebox, y_movebox = int(x_movebox), int(y_movebox)
        if x_movebox+self.l >= 2 * self.L -1:
            x_movebox -= 1
        elif x <= self.l:
            x_movebox += 1
        else:
            if np.random.rand() > 0.5:
                x_movebox -= 1
            else:
                x_movebox += 1
        self.movebox_latent_state = np.array([x_movebox, y_movebox])
        canvas[x_movebox - self.l:x_movebox + self.l + 1, y_movebox - self.l:y_movebox + self.l + 1] = 255
        self.observ = canvas
        done = (x <= self.l) or ( x+self.l >= 2 * self.L -1)
        done = bool(done)
        if not done:
            reward = -0.1
        else:
            if x <= self.l:
                reward = 10
            else:
                reward = -10
        return self.observ, reward, done, {}

    def reset(self):
        self.movebox_latent_state = np.ceil(np.ceil(np.ceil(np.random.rand(2) * (2 * self.L-self.l))+self.l))
        self.latent_state = (50,50)
        #canvas = torch.ones(2*self.L, 2*self.L)*150
        canvas = ATARI_CANVAS
        x, y = self.latent_state
        x, y = int(x), int(y)
        x_movebox, y_movebox = self.movebox_latent_state
        x_movebox, y_movebox = int(x_movebox), int(y_movebox)
        canvas[x-self.l:x+self.l+1, y-self.l:y+self.l+1] = 255
        canvas[x_movebox - self.l:x_movebox + self.l + 1, y_movebox - self.l:y_movebox + self.l + 1] = 255
        self.observ = canvas
        return self.observ

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        #x = torch.unsqueeze(x, 1) # dim(x) = (100,100)
        #x = self.convs(x)
        #x = x.view(-1, self.conv_output_size)
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class Branch(nn.Module):
    def __init__(self):
        super(Branch, self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(1, 32, 5, stride=5, padding=0), nn.ReLU(),
                                   nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
        self.conv_output_size = 576
        self.W_h = nn.Parameter(torch.rand(self.conv_output_size, 256))
        self.W_c = nn.Parameter(torch.rand(256, N_STATES))
        self.b_h = nn.Parameter(torch.zeros(256))
        self.b_c = nn.Parameter(torch.zeros(N_STATES))
        self.W = nn.Parameter(torch.rand(N_STATES, N_STATES))

    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)
        h = torch.matmul(x, self.W_h) + self.b_h  # Contrastive head
        h = nn.LayerNorm(h.shape[1])(h)
        h = F.relu(h)
        h = torch.matmul(h, self.W_c) + self.b_c  # Contrastive head
        h = nn.LayerNorm(N_STATES)(h)
        return h

class Moco(object):
    def __init__(self):
        self.online_net = Branch()
        self.momentum_net = Branch()
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=0.0001, eps=1.5e-4)
        self.observ_memory = torch.zeros((OBSERV_MEMORY_CAPACITY, 100, 100))
        self.memory_counter = 0
        self.update_counter = 0

    def store_observ(self, o):
        index = self.memory_counter % OBSERV_MEMORY_CAPACITY
        self.observ_memory[index, :] = o
        self.memory_counter += 1

    def learn(self):
        if self.update_counter % MOMENTUM_UPDATE_ITER == 0:
            self.update_momentum_net()
        self.update_counter += 1
        sample_index = np.random.choice(OBSERV_MEMORY_CAPACITY, BATCH_SIZE)
        o = self.observ_memory[sample_index, :]
        aug_o_1 = random_shift(torch.unsqueeze(torch.squeeze(o,0),1))
        aug_o_2 = random_shift(torch.unsqueeze(torch.squeeze(o,0),1))
        z_anch = self.online_net(aug_o_1)
        z_target = self.momentum_net(aug_o_2)
        z_proj = torch.matmul(self.online_net.W, z_target.T)
        logits = torch.matmul(z_anch, z_proj)
        logits = (logits - torch.max(logits, 1)[0][:, None])
        logits = logits * 0.1
        labels = torch.arange(logits.shape[0]).long()
        moco_loss = nn.CrossEntropyLoss()(logits, labels)
        self.online_net.zero_grad()
        moco_loss.mean().backward()
        clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimiser.step()
        self.update_counter += 1
        self.moco_loss = moco_loss

    def update_momentum_net(self, momentum=0.999):
        for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES*2+2))  # initialize memory
        self.state_memory = np.zeros((MEMORY_CAPACITY, N_STATES, N_STATES*2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

print('\nCollecting experience...')
env = Box2DEnv()
#env = MovingBox2DEnv()
N_ACTIONS = env.action_space.n
N_STATES = 128 # x.shape = (N_STATES, N_STATES)
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
dqn = DQN()
moco = Moco()
Ep_r_hist = []
CL_hist = []

for i_episode in range(100):
    observ = env.reset()
    ep_r = 0
    while True:
        if moco.memory_counter > OBSERV_MEMORY_CAPACITY and moco.memory_counter%20 == 0:
            moco.learn()
            CL_hist.append(moco.moco_loss)
        #print(moco.moco_loss)
        #CL_hist.append(moco.moco_loss)
        #env.render()
        moco.store_observ(observ)
        s = moco.online_net(aug.RandomCrop((84, 84))(observ)).detach().numpy().squeeze(0)
        a = dqn.choose_action(s)

        # take action
        observ_, r, done, info = env.step(a)
        moco.store_observ(observ_)
        s_ = moco.online_net(aug.RandomCrop((84, 84))(observ_)).detach().numpy().squeeze(0)
        dqn.store_transition(s, a, r, s_)
        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                if i_episode%10 == 0: print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))
                Ep_r_hist.append(ep_r)
        if done:
            break
        observ = observ_
#plt.plot(Ep_r_hist)
#plt.show()
#print(CL_hist[300:350])
#plt.savefig('grey_background_moving_box.png')
plt.plot(CL_hist)
plt.show()
