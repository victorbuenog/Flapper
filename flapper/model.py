import numpy
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import matplotlib.path as mpath
from matplotlib.markers import MarkerStyle
import matplotlib.font_manager as fm
plt.ioff() 

from .env import InitialCondition
from .env import SwimmerEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.type(torch.float32)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Ensure state_values are float32
            state_values = state_values.type(torch.float32)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


class Trainer(object):
    def __init__(self, **env_args):
        ############## Hyperparameters ##############
        self.env_name = "Flappers"
        self.env = SwimmerEnv(**env_args)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 5
        self.render = False
        self.solved_reward = 500         # stop training if avg_reward > solved_reward
        self.log_interval = 20           # print avg reward in the interval
        self.max_episodes = 1000        # max training episodes
        self.max_timesteps = 500       # max timesteps in one episode
        self.n_latent_var = 64           # number of variables in hidden layer
        self.update_timestep = 200    # update policy every n timesteps
        self.lr = 0.002
        self.betas = (0.9, 0.999)
        self.gamma = 0.99                # discount factor
        self.K_epochs = 4               # update policy for K epochs
        self.eps_clip = 0.2              # clip parameter for PPO
        self.random_seed = 4
        #############################################

        # if self.random_seed:
        #     torch.manual_seed(self.random_seed)
        #     self.env.seed(self.random_seed)

        self.memory = Memory()
        self.ppo = PPO(self.state_dim, self.action_dim, self.n_latent_var, self.lr, self.betas, self.gamma, self.K_epochs, self.eps_clip)

    def load_model(self, model_path):
        self.ppo.policy.load_state_dict(torch.load(model_path))
        self.ppo.policy_old.load_state_dict(torch.load(model_path))

    def get_policy(self, episodes=1000):
        pol_name = f"PPO_{self.env_name}_{episodes}.pth"
        return pol_name

    def train(self, initial_condition=None, initial_condition_fn=None, episodes=None):

        # logging variables
        running_reward = 0
        avg_length = 0
        timestep = 0
        history = []

        # header
        with open(f"log_PPO_{self.env_name}.csv", 'a') as f:
                    f.write(f'Episode,avg length,reward\n')
        # training loop
        if episodes is None:
            episodes = self.max_episodes
        for i_episode in range(1, episodes+1):
            state = self.env.reset(initial_condition=initial_condition, initial_condition_fn=initial_condition_fn)
            for t in range(self.max_timesteps):
                timestep += 1

                # Running policy_old:
                action = self.ppo.policy_old.act(state, self.memory)
                state, reward, done, info = self.env.step(action)

                # Saving reward and is_terminal:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)

                # update if its time
                if timestep % self.update_timestep == 0:
                    self.ppo.update(self.memory)
                    self.memory.clear_memory()
                    timestep = 0

                running_reward += reward
                if self.render:
                    self.env.render()
                if done:
                    break

                history.append(info)

            avg_length += t

            filename = f"./PPO_{self.env_name}_{i_episode}.pth"

            # save every 100 episodes
            if i_episode % 100 == 0:
                    torch.save(self.ppo.policy.state_dict(), filename)
            # logging
            if i_episode % self.log_interval == 0:
                avg_length = int(avg_length/self.log_interval)
                running_reward = ((running_reward/self.log_interval))

                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))

                # Save message on csv file
                with open(f"log_PPO_{self.env_name}.csv", 'a') as f:
                    f.write(f'{i_episode},{avg_length},{running_reward}\n')
                running_reward = 0
                avg_length = 0

        return history;

    def test_policy(self, initial_conditions):

        max_timesteps = 500

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 20))

        for ic in initial_conditions:
            obs = self.env.reset(initial_condition=ic)
            schooling_numbers = []
            flow_agreements = []
            avg_flow_agreements = []
            v_gradients = []
            u2 = []
            action_values = []
            p_values = []
            for t in range(max_timesteps):
                action = self.ppo.policy.act(obs, Memory())
                obs, reward, done, info = self.env.step(action)
                schooling_number = (info['distance']-SwimmerEnv.c)/info['u1']/info['f1']
                schooling_numbers.append(schooling_number)
                flow_agreements.append(info['flow_agreement'])
                avg_flow_agreements.append(info['avg_flow_agreement'])
                v_gradients.append(info['v_gradient'])
                u2.append(info['avg_u2'])
                p_values.append(info['avg_power'])
                if done:
                    break

            # plt.plot(distances, label=f"Initial Distance: {ic.distance}")
            ax1.plot(schooling_numbers, label=f"Initial Distance: {ic.distance}")

            ax2.plot(p_values, label=f"Initial Distance: {ic.distance}")

            ax3.plot(avg_flow_agreements, label=f"Initial Distance: {ic.distance}")

            ax4.plot(v_gradients, label=f"Initial Distance: {ic.distance}")

            ax5.plot(u2, label=f"Initial Distance: {ic.distance}")

        ax1.set_xlabel('Timesteps (0.1 s)')
        ax1.set_ylabel('Schooling Number')
        ax1.set_title('Distance Between Leader and Follower for Different Initial Conditions')
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel('Timesteps (0.1 s)')
        ax2.set_ylabel('Average Power')
        # ax2.set_xlim([0, 100])
        ax2.set_title('Power for Different Initial Conditions')
        ax2.legend()
        ax2.grid(True)

        ax3.set_xlabel('Timesteps (0.1 s)')
        ax3.set_ylabel('Average Flow Agreement')
        # ax3.set_xlim([0, 100])
        ax3.set_title('Average Flow Agreement for Different Initial Conditions')
        ax3.legend()
        ax3.grid(True)

        ax4.set_xlabel('Timesteps (0.1 s)')
        ax4.set_ylabel('Velocity Gradient')
        # ax4.set_xlim([0,100])
        std_vel_gradient = numpy.std(v_gradients)
        ax4.set_ylim([-3*std_vel_gradient,3*std_vel_gradient])
        ax4.set_title('Velocity Gradient for Different Initial Conditions')
        ax4.legend()
        ax4.grid(True)

        ax5.set_xlabel('Timesteps (0.1 s)')
        ax5.set_ylabel('Horizontal Velocity')
        # ax5.set_xlim([0,100])
        ax5.set_title('Average Velocity for Different Initial Conditions')
        ax5.legend()
        ax5.grid(True)

        plt.tight_layout()
        plt.show()

    def naca_airfoil(self, code, num_points=100):
        """Generates the coordinates of a NACA 4-digit airfoil."""
        m = float(code[0]) / 100.0  # Maximum camber
        p = float(code[1]) / 10.0  # Location of maximum camber
        t = float(code[2:]) / 100.0  # Maximum thickness

        x = numpy.linspace(0, 1, num_points)
        yt = 5 * t * (0.2969 * numpy.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

        if m == 0 and p == 0:
            yc = numpy.zeros_like(x)
            theta = numpy.zeros_like(x)
        else:
            yc = numpy.where(x <= p,
                            m / p**2 * (2 * p * x - x**2),
                            m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
            theta = numpy.arctan(numpy.gradient(yc, x))

        xu = x - yt * numpy.sin(theta)
        yu = yc + yt * numpy.cos(theta)
        xl = x + yt * numpy.sin(theta)
        yl = yc - yt * numpy.cos(theta)

        # Close the path by adding the first point to the end
        x_coords = -numpy.concatenate([xu, xl[::-1]])
        y_coords = numpy.concatenate([yu, yl[::-1]])

        return mpath.Path(numpy.column_stack([x_coords, y_coords]))

    def animate(self, i):
        font_prop = fm.FontProperties(size=35)
        self.ax.clear()
        x_leader = self.leader_positions[i][0]
        y_leader = self.leader_positions[i][1]
        x_follower = self.follower_positions[i][0]
        y_follower = self.follower_positions[i][1]
        self.ax.set_xlim([x_leader - 50, x_leader + 5])
        self.ax.set_ylim([-1.5 * self.env.A1, 1.5 * self.env.A1])
        self.ax.scatter(x_leader, y_leader, label='leader', color='black', marker=self.fish_marker)
        self.ax.scatter(x_follower, y_follower, label='follower', color='black', marker=self.fish_marker)
        self.ax.grid(True, axis='x', color='grey')
        self.ax.set_xlabel('X Position (cm)', fontproperties = font_prop)
        self.ax.set_ylabel('Y Position (cm)', fontproperties = font_prop)
        self.ax.tick_params(axis='both', which='major', labelsize=30)

        # Display wake
        T = self.env.T
        for j, wake_pos in enumerate(self.leader_positions[:i]):
            t_delay = (i - j) / self.frame_rate
            wake_amplitude_scale = numpy.exp(-t_delay / T)
            self.ax.scatter(wake_pos[0], wake_pos[1]*wake_amplitude_scale, color='blue', s=5, marker='o')

        return self.ax

    def create_video(self, ic=InitialCondition(distance=30, f2=1.), time=10):
        state = self.env.reset(ic)

        airfoil_path = self.naca_airfoil("0017")
        self.fish_marker = MarkerStyle(airfoil_path, transform=mpl.transforms.Affine2D().scale(16))

        self.leader_positions = []
        self.follower_positions = []
        self.frame_rate = 12
        runtime = time # seconds

        for _ in range(self.frame_rate*runtime):
            action = self.ppo.policy.act(state, Memory())
            state, reward, done, info = self.env.step(action)

            self.leader_positions.append((info['x1'], info['y1']))
            self.follower_positions.append((info['x2'], info['y2']))

            if done:
                break

        fig, self.ax = plt.subplots(figsize=(21, 9))
        ani = animation.FuncAnimation(fig, self.animate, frames=len(self.leader_positions), interval=self.frame_rate, blit=False)

        # To save the animation as a video file:
        ani.save('swimmer_animation.mp4', writer='ffmpeg', fps=self.frame_rate)

        # Plot distances
        distances = [numpy.linalg.norm(numpy.array(leader_pos) - numpy.array(follower_pos))
                    for leader_pos, follower_pos in zip(self.leader_positions, self.follower_positions)]
        times = numpy.arange(len(distances)) * (1.0 / self.frame_rate)

        plt.figure()
        plt.plot(times, distances)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Distance')
        plt.title('Distance between Leader and Follower vs. Time')
        plt.grid(True)
        plt.show()

        return;