import numpy as np


from mlagents_envs.environment import UnityEnvironment
env = UnityEnvironment(file_name = 'Road1/Prototype 1')

env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]

t1 = 0
t2 = 0
dg = 0
act = 0

for i in range(10000):
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    print("\n\n L : ", cur_obs[9])
    print("FL : ", cur_obs[8])
    print("FM : ", cur_obs[10])
    print("FR : ", cur_obs[6])
    print(" R : ", cur_obs[7])

    # x = cur_obs[0], z = cur_obs[2]
    # s1 = cur_obs[6], s2 = cur_obs[7], s3 = cur_obs[8], s4 = cur_obs[9], s5 = cur_obs[10]

    # Set the actions

    if (t1 == 0 and t2 == 0):
        env.set_actions(behavior_name, np.array([[0,150,150]]))
        act = 0

        if (cur_obs[9] > 18):
            env.set_actions(behavior_name, np.array([[-0.3, 150, 150]]))
            act = 1
        if (cur_obs[7] > 18):
            env.set_actions(behavior_name, np.array([[0.3, 150, 150]]))
            act = 1

        if (cur_obs[6] < cur_obs[8] and cur_obs[6] < 7):
            env.set_actions(behavior_name, np.array([[-0.5, 150, 150]]))
            act = 2
        if (cur_obs[6] > cur_obs[8] and cur_obs[8] < 7):
            env.set_actions(behavior_name, np.array([[0.5, 150, 150]]))
            act = 2

        if (cur_obs[7] < 4):
            env.set_actions(behavior_name, np.array([[-0.3, 150, 150]]))
            t1 = 3
            t2 = 3
            dg = -0.3
            act = 3
        if (cur_obs[9] < 4):
            env.set_actions(behavior_name, np.array([[0.3, 150, 150]]))
            t1 = 3
            t2 = 3
            dg = 0.3
            act = 3

        if ((cur_obs[6] < 8 or cur_obs[8] < 8 or cur_obs[10] < 8)
                and cur_obs[6] + cur_obs[7] > cur_obs[8] + cur_obs[9]):
            env.set_actions(behavior_name, np.array([[0.7, 150, 150]]))
            t1 = 5
            t2 = 5
            dg = 0.7
            act = 4
        if ((cur_obs[6] < 8 or cur_obs[8] < 8 or cur_obs[10] < 8)
                and cur_obs[6] + cur_obs[7] < cur_obs[8] + cur_obs[9]):
            env.set_actions(behavior_name, np.array([[-0.7, 150, 150]]))
            t1 = 5
            t2 = 5
            dg = -0.7
            act = 4

        if (cur_obs[8] < cur_obs[10] and cur_obs[10] < cur_obs[6] and cur_obs[6] < 12
                and cur_obs[9] < 4.5 and cur_obs[7] > 19):
            env.set_actions(behavior_name, np.array([[1.0, 150, 150]]))
            t1 = 10
            t2 = 10
            dg = 1.0
            act = 5

    if (t1 == 0 and t2 != 0):
        t2 = t2 - 1
        act = 9
        env.set_actions(behavior_name, np.array([[0, 150, 150]]))

    if (t1 != 0):
        t1 = t1 - 1
        env.set_actions(behavior_name, np.array([[dg, 150, 150]]))

    print("\nAction : ", act, "   Range : ", i)

    # Move the simulation forward
    env.step()

env.close()
