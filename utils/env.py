import gym
from gym_minigrid.wrappers import *
from multiprocessing import Process, Pipe
from gym.wrappers import AtariPreprocessing, TransformReward, RescaleAction
from gym import ObservationWrapper
from gym.spaces import Box 


def make_env(env_key, seed=None):
    # create env
    env = gym.make(env_key, full_action_space=False, difficulty=3, frameskip=1)

    env = AtariPreprocessing(env, screen_size=84, terminal_on_life_loss=True, scale_obs=True, grayscale_newaxis=True)
    
    #env = LimitSpace(env)
    
    env = TransformReward(env, lambda r: r / 400)
    
    env = ActionWrapper(env)   
    
   
    # ograniczenie akcji
    # env = RescaleAction(env, min_action=1, max_action=6)
    env.action_space.n = 5    
    env.action_space.start = 1 
    
    env.reset(seed=seed)
    return env

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step([data])
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, act):
        new_act=act+1
        return new_act       
class LimitSpace(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(shape=(68, 68), low=0, high=1) # self.observation_space = Box(shape=(64, 64), low=0, high=1), shape 66 66

    def observation(self, obs):
        return obs[3:71, 8:76, :] # [4:70, 9:75, :] obs[5:69, 10:74, :]
class ParallelEnv(gym.Env):
    """
    A concurrent execution of environments in multiple processes.
    """

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step([actions[0]])
        if done:
            obs = self.envs[0].reset()
        results = zip(
            *[(obs, reward, done, info)] + [local.recv() for local in self.locals]
        )
        return results

    def render(self):
        raise NotImplementedError
