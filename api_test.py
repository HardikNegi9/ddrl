from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from stable_baselines3 import PPO
import gym

from env import DeadlockRecoveryEnv

app = FastAPI()

class MatrixInput(BaseModel):
    allocation: List[List[int]]
    request: List[List[int]]
    available: List[int]

class SimulationStep(BaseModel):
    step: int
    action: str
    available: Optional[List[int]] = None
    allocation: Optional[List[List[int]]] = None
    request: Optional[List[List[int]]] = None
    work: Optional[List[int]] = None
    finish: List[bool]
    result: Optional[str] = None
    final_finish: Optional[List[bool]] = None

class SimulationResult(BaseModel):
    simulation_id: str
    steps: List[SimulationStep]

model = PPO.load("ppo_deadlock_multi_env")  # Load trained model

@app.post("/check_deadlock", response_model=SimulationResult)
def check_deadlock(matrix_input: MatrixInput):
    allocation = np.array(matrix_input.allocation)
    request = np.array(matrix_input.request)
    available = np.array(matrix_input.available)

    num_processes = len(allocation)
    env = DeadlockRecoveryEnv(num_processes=num_processes, num_resources=len(available))
    obs = env.reset(allocation=allocation, request=request, available=available)

    steps = []
    step_count = 0
    done = False

    while not done:
        step_info = SimulationStep(
            step=step_count,
            action=f"Step {step_count} executed",
            allocation=env.allocation.tolist(),
            request=env.request.tolist(),
            available=env.available.tolist(),
            finish=[bool(np.all(env.request[i] == 0)) for i in range(env.num_processes)]
        )
        steps.append(step_info)

        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        step_count += 1

    steps.append(SimulationStep(
        step=step_count,
        action="Deadlock Check Completed",
        result="No Deadlock" if not env._is_deadlocked() else "Deadlock Detected",
        final_finish=[bool(np.all(env.request[i] == 0)) for i in range(env.num_processes)],
        finish=[bool(np.all(env.request[i] == 0)) for i in range(env.num_processes)]
    ))

    return SimulationResult(
        simulation_id="matrix_sim",
        steps=steps
    )
