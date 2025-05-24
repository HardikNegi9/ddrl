# deadlock_env.py

import gym
from gym import spaces
import numpy as np

class DeadlockRecoveryEnv(gym.Env):
    def __init__(self, num_processes=10, num_resources=5):
        super(DeadlockRecoveryEnv, self).__init__()
        self.num_processes = num_processes
        self.num_resources = num_resources

        self.action_space = spaces.Discrete(2 * self.num_processes + 1)
        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(2 * self.num_processes * self.num_resources + self.num_resources,),
            dtype=np.int32
        )

        self.reset()

    def reset(self, allocation=None, request=None, available=None):
        if allocation is not None:
            self.allocation = np.array(allocation)
        else:
            self.allocation = np.random.randint(0, 3, (self.num_processes, self.num_resources))

        if request is not None:
            self.request = np.array(request)
        else:
            self.request = np.random.randint(0, 3, (self.num_processes, self.num_resources))

        if available is not None:
            self.available = np.array(available)
        else:
            self.available = np.random.randint(3, 7, self.num_resources)

        self.steps = 0
        self.last_action = None
        self.last_killed = None
        self.last_preempted = []

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.allocation.flatten(),
            self.request.flatten(),
            self.available
        ])

    def step(self, action):
        reward = 0
        done = False
        self.last_action = action
        self.last_killed = None
        self.last_preempted = []

        if self._is_deadlocked():
            if action == 0:
                reward = -100
            else:
                reward = -10
                self._kill_process(action - 1)
                self.last_killed = action - 1
        else:
            reward = 1
            for i in range(self.num_processes):
                if any(self.request[i] > 0) and np.any(self.available > 0):
                    self._preempt_request(i)
                    self.last_preempted.append(i)

        self.steps += 1
        if self.steps >= 20:
            done = True

        return self._get_obs(), reward, done, {}

    def _kill_process(self, pid):
        self.available += self.allocation[pid]
        self.allocation[pid] = 0
        self.request[pid] = 0

    def _preempt_request(self, pid):
        preempted = np.minimum(self.request[pid], np.ones_like(self.request[pid]))
        self.available += preempted
        self.request[pid] -= preempted

    def _is_deadlocked(self):
        work = self.available.copy()
        finish = [False] * self.num_processes

        changed = True
        while changed:
            changed = False
            for i in range(self.num_processes):
                if not finish[i] and np.all(self.request[i] <= work):
                    work += self.allocation[i]
                    finish[i] = True
                    changed = True

        return not all(finish)

    def render(self, mode='human'):
        print("\n==========================")
        print(f"Step: {self.steps}")
        print("Process Status:")
        for i in range(self.num_processes):
            print(f"  P{i}: Allocated: {self.allocation[i]}, Requesting: {self.request[i]}")
        print(f"Available Resources: {self.available}")

        if self._is_deadlocked():
            print("⚠️  Deadlock detected.")
        else:
            print("✅  System is in a safe state.")

        if self.last_killed is not None:
            print(f"❌ Process P{self.last_killed} was killed.")
        if self.last_preempted:
            for pid in self.last_preempted:
                print(f"⚡ Process P{pid} had resources preempted.")
        print("==========================\n")
