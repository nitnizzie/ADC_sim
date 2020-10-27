import numpy as np


from mlagents_envs.environment import UnityEnvironment
env = UnityEnvironment(file_name = 'Prototype 1')

env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]

t = 0
a1 = 0
a2 = 150
a3 = 150

for i in range(10000):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    print("cur observations : ", decision_steps.obs[0][0,:])

    # x = cur_obs[0], z = cur_obs[2]
    # s1 = cur_obs[6], s2 = cur_obs[7], s3 = cur_obs[8], s4 = cur_obs[9], s5 = cur_obs[10]

    # Set the actions
    if (t == 0):
        env.set_actions(behavior_name, np.array([[0,150,150]]))

        if (cur_obs[9] > 15):
            env.set_actions(behavior_name, np.array([[-0.3, 150, 150]]))
        if (cur_obs[7] > 15):
            env.set_actions(behavior_name, np.array([[0.3, 150, 150]]))

        if (cur_obs[6] < cur_obs[8] and cur_obs[6] < 15):
            env.set_actions(behavior_name, np.array([[-0.5, 150, 150]]))
        if (cur_obs[6] > cur_obs[8] and cur_obs[8] < 15):
            env.set_actions(behavior_name, np.array([[0.5, 150, 150]]))

        if (cur_obs[7] < 5):
            env.set_actions(behavior_name, np.array([[-0.5, 150, 150]]))
            t = 3
            a1 = -0.5
        if (cur_obs[9] < 5):
            env.set_actions(behavior_name, np.array([[0.5, 150, 150]]))
            t = 3
            a1 = 0.5

    else :
        t = t - 1
        env.set_actions(behavior_name, np.array([[a1, a2, a3]]))

    # Move the simulation forward
    env.step()

env.close()
