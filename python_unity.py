import numpy as np


from mlagents_envs.environment import UnityEnvironment
env = UnityEnvironment(file_name = 'Road1/Prototype 1')

env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]
for i in range(100):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    print("cur observations : ", decision_steps.obs[0][0,:])
    # Set the actions
    env.set_actions(behavior_name, np.array([[0,150,150]]))
    # Move the simulation forward
    env.step()

env.close()
