import random
import numpy
import scipy
import matplotlib.pyplot as plt
plt.ioff() 
import gym
from gym import wrappers
from gym import spaces

import imageio

class InitialCondition(object):

    def __init__(self, distance=None, f2=None, A2=None, goal=None):
        self.distance = distance if distance is not None else 21.5
        self.A1 = 2.0
        self.f1 = 1.0
        self.A2 = 2.0 if A2 is None else A2
        self.f2 = 1.0 if f2 is None else f2
        self.goal = goal if goal is not None else 21.5

        self.u2 = numpy.pi * self.A2 * self.f2 * numpy.sqrt(2 * SwimmerEnv.Ct / SwimmerEnv.Cd)
        self.v2 = -self.A2*(2 * numpy.pi * self.f2)
        self.t_delay = -self.distance/self.u2
        self.v_flow = self.A1*self.f1*numpy.cos(2*numpy.pi*self.f1*(-self.t_delay))*numpy.exp(-self.t_delay/SwimmerEnv.T)
        self.flow_agreement = self.v2 * self.v_flow

    def random(self, randomize_fields=[]):
        if 'distance' in randomize_fields:
            self.distance = random.uniform(10,30)
        if 'f2' in randomize_fields:
            self.f2 = random.uniform(0.5, 1.5)
        if 'A2' in randomize_fields:
            self.A2 = random.uniform(.5, 3.0)
        if 'v2' in randomize_fields:
            self.v2 = random.uniform(-1.0, 1.0)
        return self


class SwimmerEnv(gym.Env):

    s = 15.
    c = 4.
    As = s * c
    T = 5
    m = 80.
    Ct = .96
    Cd = .25
    rho = 1.
    dt = 0.1

    def __init__(self, rewards=[]):
        super(SwimmerEnv, self).__init__()

        self.action_space = spaces.Discrete(5) # Increase, decrease, or maintain value for f2 and A2
        self.n_history = 50 # observation timesteps

        # Observations: f2, A2, distance, v2-v_flow, u2
        obs_low = [-numpy.inf for _ in range(5*self.n_history)]
        obs_high = [numpy.inf for _ in range(5*self.n_history)]

        self.observation_space = spaces.Box(low=numpy.array(obs_low), high=numpy.array(obs_high), dtype=numpy.float32)
        self.rewards = rewards

        self.flap = None
        self.t_bound = 500.

        self.reset()

    def _shoot(self, A1, A2, f1, f2, vec_initial, t_start=0., t_bound=5000., method='RK45'):
        rho = self.rho
        As = self.As
        T = self.T
        m = self.m
        Ct = self.Ct
        Cd = self.Cd
        s = self.s
        c = self.c

        def fun(t, vec):
            x2, u2 = vec
            u1 = numpy.pi * A1 * f1 * numpy.sqrt(2 * Ct / Cd)  # Leader velocity (constant)
            dt = -x2 / u1 + t
            self.Ft2 = 2*rho*As*Ct*numpy.pi**2*((A2*f2*numpy.cos(2*numpy.pi*f2*t)-A1*f1*numpy.cos(2*numpy.pi*f1*(t-dt))*numpy.exp(-dt/T)))**2
            self.Fd2 = rho*As*Cd*u2**2/2
            dy_dt = (u2, (self.Ft2 - self.Fd2)/m)
            return numpy.asarray(dy_dt)
        # events = [lambda t, y: y[0] - y[4] - 0.00001]
        events = []
        for ee in events: setattr(ee, 'terminal', True)
        solver = scipy.integrate.solve_ivp(fun, (t_start, t_bound), vec_initial, method=method, events=events,
                                            rtol=1e-4, atol=1e-7, max_step=.03, first_step=.001, dense_output=True)
        u1 = numpy.pi * A1 * f1 * numpy.sqrt(2 * Ct / Cd)  # Leader velocity (constant)
        x1 = u1 * solver.t[-1]  # Leader x-coord (calculated)
        y1 = A1 * numpy.sin(2 * numpy.pi * f1 * solver.t[-1])  # Leader y-coord (calculated)
        v1 = -A1*(2 * numpy.pi * f1)*numpy.cos(2 * numpy.pi * f1 * solver.t[-1])
        y2 = A2 * numpy.sin(2 * numpy.pi * f2 * solver.t[-1])  # Leader y-coord (calculated)
        v2 = -A2*(2 * numpy.pi * f2)*numpy.cos(2 * numpy.pi * f2 * solver.t[-1])
        values = list(zip(solver.t, solver.y.T))

        info = {
            'x1': x1,
            'y1': y1,
            'u1': u1,
            'v1': v1,
            'x2': values[-1][1][0],
            'y2': y2,
            'u2': values[-1][1][1],
            'v2': v2,
        }
        return solver, values, info


    def _get_obs(self):
        obs = [
            self.f2,
            self.A2,
            self.distance,
            self.v2 - self.v_flow,
            self.avg_u2,
        ]

        return numpy.array(obs, dtype=numpy.float32)


    def step(self, action):
        Ct = self.Ct
        Cd = self.Cd

        match action:
            case 0:
                self.f2 = max(self.f2 - 0.1, 0.5)
            case 1:
                self.f2 = min(self.f2 + 0.1, 1.5)
            case 2:
                self.A2 = max(self.A2 - 0.1, 0.5)
            case 3:
                self.A2 = min(self.A2 + 0.1, 3.0)
            case 4: # No change
                self.A2 = self.A2
                self.f2 = self.f2

        t_bound_step = self.dt
        solver, values, shoot_info = self._shoot(self.A1, self.A2, self.f1, self.f2, self.flap, t_start=self.tt, t_bound=self.tt+t_bound_step)
        self.flap = values[-1][1]
        self.tt += self.dt
        self.u1 = numpy.pi * self.A1 * self.f1 * numpy.sqrt(2 * Ct / Cd)  # Leader velocity
        self.x1 = self.u1 * self.tt  # Leader x-coord
        self.y1 = self.A1 * numpy.sin(2 * numpy.pi * self.f1 * self.tt)  # Leader y-coord
        self.y2 = self.A2 * numpy.sin(2 * numpy.pi * self.f2 * self.tt)  # Follower y-coord
        self.v1 = -self.A1*(2 * numpy.pi * self.f1)*numpy.cos(2 * numpy.pi * self.f1 * self.tt)
        self.t_delay = self.tt - self.flap[0]/self.u1
        v_flow = self.A1*self.f1*numpy.cos(2*numpy.pi*self.f1*(self.tt-self.t_delay))*numpy.exp(-self.t_delay/self.T)
        self.t_delay_head = self.tt - (self.flap[0]+self.c)/self.u1
        v_flow_head = self.A1*self.f1*numpy.cos(2*numpy.pi*self.f1*(self.tt-self.t_delay_head))*numpy.exp(-self.t_delay_head/self.T)
        v_gradient = (v_flow_head - v_flow)/self.c
        self.u2 = self.flap[1]

        alpha = -numpy.arctan2(shoot_info['v2']-v_flow, self.u2)
        Cl = 2*numpy.pi*numpy.sin(alpha)
        v2_dot = (shoot_info['v2'] - self.previous_v2)/self.dt
        # L = Cl*self.rho*self.As*self.u2**2/2
        # Fy = self.m*v2_dot - L*numpy.cos(-alpha)
        L = 1*self.rho*self.As*(shoot_info['v2']-v_flow)**2/2
        Fy = self.m*v2_dot - L
        self.power = (Fy*shoot_info['v2']) / 1e5

        self.distance = self.x1 - self.flap[0]-self.c  # Distance between leader and follower
        done = False
        new_distance_from_goal = numpy.abs(self.distance - self.goal) # Distance
        new_flow_agreement = shoot_info['v2'] * v_flow_head # Flow agreement
        a_flow = (v_flow-self.previous_v_flow)/self.dt # Flow acceleration


        # Calculate average flow agreement over one period
        self.flow_agreement_history.append(new_flow_agreement)
        if len(self.flow_agreement_history) > 10:
            self.flow_agreement_history.pop(0)
        new_avg_flow_agreement = numpy.mean(self.flow_agreement_history)

        # Calculate average follower velocity over one period
        self.u2_history.append(self.u2)
        if len(self.u2_history) > 10:
            self.u2_history.pop(0)
        self.avg_u2 = numpy.mean(self.u2_history)

        # Power average
        self.power_history.append(self.power)
        if len(self.power_history) > 10:
            self.power_history.pop(0)
        self.avg_power = numpy.mean(self.power_history)

        # distance average
        self.distance_history.append(self.distance)
        if len(self.distance_history) > self.n_history:
            self.distance_history = self.distance_history[-self.n_history:]
        self.avg_distance = numpy.mean(self.distance_history)

        # Reset reward
        self.reward = 0.

        # Penalize change in distance
        self.reward -= 0.001*abs(self.avg_distance - self.distance)

        # Penalize extreme behaviors
        if self.distance < 0. or self.distance > 200:
            self.reward -= 100
            done = True

        self.previous_power = self.avg_power
        self.previous_v_flow = v_flow
        self.distance_from_goal = new_distance_from_goal
        self.flow_agreement = new_flow_agreement
        self.avg_flow_agreement = new_avg_flow_agreement
        self.last_reward = self.reward
        self.previous_v2 = shoot_info['v2']

        # Update observation history
        self.obs_history.append(self._get_obs())
        if len(self.obs_history) > self.n_history:
            self.obs_history.pop(0)
        history_obs = numpy.array(self.obs_history).flatten()

        info = {
            'distance': self.distance,
            'action': action,
            'reward': self.reward,
            'done': done,
            'f1': self.f1,
            'f2': self.f2,
            'A2': self.A2,
            'distance_from_goal': self.distance_from_goal,
            't': self.tt,
            'v_flow': v_flow,
            'v_flow_head': v_flow_head,
            'a_flow': a_flow,
            'flow_agreement': new_flow_agreement,
            'avg_flow_agreement': new_avg_flow_agreement,
            'u2': self.flap[1],
            'avg_u2': self.avg_u2,
            'v_gradient': v_gradient,
            'avg_power': self.avg_power,
        }
        info.update(shoot_info)

        return history_obs, self.reward, done, info

    def reset(self, initial_condition=None, initial_condition_fn=None):
        Ct = self.Ct
        Cd = self.Cd
        gap_distance_in_wavelengths = 1.

        # get initial condition
        if initial_condition is None:
            if initial_condition_fn is not None:
                initial_condition = initial_condition_fn()
            else:
                initial_condition = InitialCondition()

        self.A1 = initial_condition.A1
        self.f1 = initial_condition.f1
        self.A2 = initial_condition.A2
        self.f2 = initial_condition.f2
        self.goal = initial_condition.goal

        u2_initial = initial_condition.u2

        self.distance = initial_condition.distance
        self.distance_from_goal = numpy.abs(self.distance - self.goal)
        self.flow_agreement = initial_condition.flow_agreement
        self.flap = numpy.asarray([-initial_condition.distance, u2_initial])
        self.tt = 0
        self.previous_v_flow = 0
        self.flow_agreement_history = []
        self.u2_history = []
        self.power_history = []
        self.previous_x2 = -self.distance
        self.flow_agreement = initial_condition.flow_agreement
        self.avg_flow_agreement = 0
        self.u2 = .1
        self.avg_u2 = .1
        self.avg_power = .1
        self.previous_power = .1
        self.previous_v2 = .1
        self.v2 = initial_condition.v2
        self.v_flow = 0
        self.distance_history = [self.distance] * self.n_history

        self.obs_history = [self._get_obs() for _ in range(self.n_history)]

        history_obs = numpy.array(self.obs_history).flatten()

        return history_obs

    def render(self, mode='rgb_array'):
        fig, ax = plt.subplots()
        ax.scatter(self.x1, self.y1, label='leader')
        ax.scatter([self.flap[0]], [self.flap[1]], label='follower')
        ax.set_xlim([self.x1 - 50, self.x1 + 5])
        ax.set_ylim([-2*self.A1, 2*self.A1])
        ax.legend()
        fig.canvas.draw()
        fig.canvas.tostring_rgb()
        frame = numpy.frombuffer(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return frame

    def close(self):
        pass