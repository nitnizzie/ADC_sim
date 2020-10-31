import math

import numpy as np
from mlagents_envs.environment import UnityEnvironment

env = UnityEnvironment(file_name = 'Road2/Prototype 1')

class decider:
    def __init__(self,env,stir_range,min_torque,max_torque):
        self.env = env
        self.stir_range = stir_range
        self.min_torque = min_torque
        self.max_torque = max_torque

        self.LR_threshold = 5
        self.FLR_threshold = 5
        self.FM_threshold = 5
        self.LR_threshold_record_range = 5
        self.FLR_threshold_record_range = 5
        self.sensor_diff = 3
        self.L_temp = []
        self.R_temp= []
        self.FL_temp = []
        self.FM_temp = []
        self.FR_temp = []

        self.small_obstacle_tuple_list = []
        self.L_general_obstacle_tuple = []
        self.R_general_obstacle_tuple = []
        self.FL_general_obstacle_tuple = []
        self.FR_general_obstacle_tuple = []

        #Before Step x,y,z coordinate
        self.xb = -900
        self.yb = -900
        self.zb = -900

        self.stop_stir_sum = 0
        self.back_stir = 0
        self.back = False
        self.stop = 0
        self.stop_count = 0
        self.stir_const = 0.7
        self.remove_distance = 3
        return

    def distance(self,x,y,z,xx,yy,zz):
        return math.sqrt((x-xx)**2 + (y-yy)**2 + (z-zz)**2)

    def decide_movement(self,x,y,z,xp,yp,zp,s1,s2,s3,s4,s5,b_stir):
        print("### DECIDE_MOVEMENT_FUNCTION ###")
        if (self.xb == -900 and self.yb == -900 and self.zb == -900):
            dx = 0
            dy = 0
            dz = 0

        else:
            dx = x - self.xb
            dy = y - self.yb
            dz = z - self.zb

        if (abs(dx) > 10 or abs(dy)>10 or abs(dz)>10):
            print("Get Back to Start Point.")
            #Change Parameters for Stirring

            #initialize dx, dy, dz
            dx = 0
            dy = 0
            dz = 0

        len_L = s4
        len_FL = s3
        len_FM = s5
        len_FR = s1
        len_R = s2

        print("X:",x,"/Z:",z)
        print("L:",len_L, "/FL:",len_FL,"/FM:",len_FM,"/FR:",len_FR,"/R:",len_R)
        dir_vec_of_car = (0,0)
        if dx != 0 and dz!=0:
            dir_vec_of_car = ((dx/math.sqrt(dx**2 + dz**2)),(dz/math.sqrt(dx**2 + dz**2)))
        print("dir_vec of car : ", dir_vec_of_car)

        toend_len = math.sqrt((xp-x)*(xp-x) + (zp-z)*(zp-z))

        #Have to direct dir_toend. However, it's just for data_collection
        dir_toend = ((xp-x)/toend_len, (zp-z)/toend_len)

        #Normal-Status Speed Equation.
        sensor_diff = self.sensor_diff
        FL = len_FL - sensor_diff; FM = len_FM - sensor_diff; FR = len_FR - sensor_diff
        a = (FL+FM+FR)/FL; b = (FL+FM+FR)/FM; c = (FL+FM+FR)/FR;
        ratio_FL = a/(a+b+c)
        ratio_FM = b/(a+b+c)
        ratio_FR = c/(a+b+c)
        print("RATIO = ",ratio_FL,":",ratio_FM,":",ratio_FR)
        tor_ave = 150*(FL*ratio_FL + FM*ratio_FM +FR*ratio_FR )/20-sensor_diff
        #Old-Speed Equation
        #tor_ave = (len_FL*50+len_FM*50+len_FR*50)/20

        #First, Check all the sensors that have values lower than threshold.
        #L_sensor
        if len_L < self.LR_threshold:
            #Left Sensor Detected Danger Situation.
            #---(1)Big-Continuous object : Just avoid it, simply.
            #---(2)Small-Dot object : This has specific crash pattern that Appear->Disappear->Not avoided->Crash!
            self.L_temp.append(len_L)
        else:
            if len(self.L_temp)<4 and len(self.L_temp)>0:
                dis_to_obs = sum(self.L_temp)/len(self.L_temp)
                self.small_obstacle_tuple_list.append(("L",dis_to_obs,x,y,z))
            self.L_temp = []

        if len(self.L_temp) > 10:
            self.L_temp = []

        #R_sensor
        if len_R < self.LR_threshold:
            self.R_temp.append(len_R)
        else:
            if len(self.R_temp)<4 and len(self.R_temp)>0:
                dis_to_obs = sum(self.R_temp)/len(self.R_temp)
                self.small_obstacle_tuple_list.append(("R",dis_to_obs,x,y,z))
            self.R_temp = []
        if len(self.R_temp) > 10:
            self.R_temp = []
        #FL_sensor
        if len_FL < self.FLR_threshold:
            self.FL_temp.append(len_FL)
        else:
            if len(self.FL_temp) < 4 and len(self.FL_temp) > 0:
                dis_to_obs = sum(self.FL_temp) / len(self.FL_temp)
                self.small_obstacle_tuple_list.append(("FL", dis_to_obs,x,y,z))
            self.FL_temp = []
        if len(self.FL_temp) > 10:
            self.FL_temp = []
        #FR_sensor
        if len_FR < self.FLR_threshold:
            self.FR_temp.append(len_FR)
        else:
            if len(self.FR_temp) < 4 and len(self.FR_temp) > 0:
                dis_to_obs = sum(self.FR_temp) / len(self.FR_temp)
                self.small_obstacle_tuple_list.append(("FR", dis_to_obs,x,y,z))
            self.FR_temp = []
        if len(self.FR_temp) > 10:
            self.FR_temp = []
        #FM sensor
        if len_FM < self.FM_threshold:
            self.FM_temp.append(len_FR)
        else:
            if len(self.FM_temp) < 4 and len(self.FM_temp) > 0:
                dis_to_obs = sum(self.FM_temp) / len(self.FM_temp)
                self.small_obstacle_tuple_list.append(("FM", dis_to_obs,x,y,z))
            self.FM_temp = []
        if len(self.FM_temp) > 10:
            self.FM_temp = []

        # if things in small obstacle value list have distance more than 3 to present position, then remove that obstacle info.
        # without above if, then th car will always get back.
        temp_list = []
        for i in self.small_obstacle_tuple_list:
            if self.distance(x,y,z,i[2],i[3],i[4]) < self.remove_distance:
                temp_list.append((i[0],i[1],i[2],i[3],i[4]))
        self.small_obstacle_tuple_list = temp_list

        #Small Obstacle Information Collection Complete.
        #Impossible Dir by Present Sensor Value + Impossible Dir by Small Obstacles.
        #Get biggest sensor value -> Go to that direction.
        #if all the sensor values(and small obstacle values) are smaller than threshold, then go back slightly.

        #Declaration
        L_min = len_L
        FL_min = len_FL
        FM_min = len_FM
        FR_min = len_FR
        R_min = len_R

        L_min_small = 20
        for i in self.small_obstacle_tuple_list:
            if i[0] == "L":
                L_min_small = min([i[1],L_min_small])

        FL_min_small = 20
        for i in self.small_obstacle_tuple_list:
            if i[0] == "FL":
                FL_min_small = min([i[1],FL_min_small])

        FM_min_small = 20
        for i in self.small_obstacle_tuple_list:
            if i[0] == "FM":
                FM_min_small = min([i[1],FM_min_small])

        FR_min_small = 20
        for i in self.small_obstacle_tuple_list:
            if i[0] == "FR":
                FR_min_small = min([i[1],FR_min_small])

        R_min_small = 20
        for i in self.small_obstacle_tuple_list:
            if i[0] == "R":
                R_min_small = min([i[1],R_min_small])

        L_min = min([L_min,L_min_small])
        FL_min = min([FL_min,FL_min_small])
        FM_min = min([FM_min,FM_min_small])
        FR_min = min([FR_min,FR_min_small])
        R_min = min([R_min,R_min_small])
        print("L_min : ",L_min)
        print("FL_min : ", FL_min)
        print("FM_min : ", FM_min)
        print("FR_min : ", FR_min)
        print("R_min : ", R_min)
        max_value = max([L_min,FL_min,FM_min,FR_min,R_min])

        print("SMALLS:", self.small_obstacle_tuple_list)
        #Go back part
        if self.stop < 1:
            self.back = False
            self.stop_stir_sum = 0
            self.back_stir = 0
            self.stop = 0
        if L_min<self.LR_threshold and R_min<self.LR_threshold and FL_min<self.FLR_threshold and FR_min<self.FLR_threshold and FM_min< self.FLR_threshold:
            self.back = True
            print("GO BACK START(Reason : STUCK!(sensor value)")
            self.stop = 25
            return 0,-100
        if self.back:
            print("GO BACK")
            self.stop-=1
            return self.back_stir,-100
        if math.sqrt(dx**2 + dz**2)<0.1:
            self.stop_stir_sum+=b_stir
            self.stop_count+=1
        if self.stop_count>20:
            self.stop = 25
            self.back = True
            self.stop_count = 0
            self.back_stir = -self.stop_stir_sum
            print("GO BACK START(Reason : STUCK!(slow)")

        #Go to the best-avoid-route.
        sel_list = [False,False,False,False,False]
        sel = 0
        if L_min == max_value:
            sel_list[0] = True
            sel-=2
            print("L have been candidate")

        if FL_min == max_value:
            sel_list[1] = True
            sel-=1
            print("FL have been candidate")

        if FR_min == max_value:
            sel_list[3] = True
            print("FR have been candidate")

        if FM_min == max_value:
            sel_list[2] = True
            sel+=1
            print("FM have been candidate")
        if R_min == max_value:
            sel_list[4] = True
            sel+=2
            print("R have been candidate")


        print(sel_list)
        cnt = 0
        for i in sel_list:
            if i:
                cnt+=1
        stir = 0

        if cnt==1:
            if sel_list[0]:
                stir = -1* self.stir_const
            elif sel_list[1]:
                stir = -0.5* self.stir_const
            elif sel_list[2]:
                stir = 0 * self.stir_const
            elif sel_list[3]:
                stir = 0.5* self.stir_const
            elif sel_list[4]:
                stir = 1* self.stir_const

        # Minimum of cnt will be 1, so else means only 2 or more.
        else:
            if sel_list[0] and sel_list[1] and sel_list[3] and sel_list[4]:
                if sel_list[2]:
                    stir = 0
                else:
                    stir = -1* self.stir_const
            if L_min < self.LR_threshold and L_min == min([L_min,FL_min,FR_min,R_min]):
                stir = 1 * self.stir_const
                print("TIE BREAKER : L")
            elif FL_min < self.FLR_threshold and FL_min == min([L_min,FL_min,FR_min,R_min]):
                stir = 0.5 * self.stir_const
                print("TIE BREAKER : FL")
            elif FR_min < self.FLR_threshold and FR_min == min([L_min,FL_min,FR_min,R_min]):
                stir = -0.5 * self.stir_const
                print("TIE BREAKER : FR")
            elif R_min < self.LR_threshold and R_min == min([L_min,FL_min,FR_min,R_min]):
                stir = -1 * self.stir_const
                print("TIE BREAKER : R")
            elif True:
                if sel == 0:
                    stir = 0
                if sel == -1:
                    stir = -0.5 * self.stir_const
                if sel == -2:
                    stir = -1 * self.stir_const
                if sel == 1:
                    stir = 0.5 * self.stir_const
                if sel == 2:
                    stir = 1 * self.stir_const
                if sel == 0 and not(sel_list[2]):
                    if sel_list[0] and sel_list[1]:
                        stir = -1* self.stir_const
                    if sel_list[3] and sel_list[4]:
                        stir = 1 * self.stir_const
                    if sel_list[0] and sel_list[3] or sel_list[0] and sel_list[4]:
                        stir = -1* self.stir_const
                    if sel_list[1] and sel_list[3]:
                        stir = -0.5*self.stir_const
                    if sel_list[3] and sel_list[4]:
                        stir = 1*self.stir_const
                    #Default Value
                    if sel_list[0]:
                        stir = -1* self.stir_const
                    if sel_list[1]:
                        stir = -0.5* self.stir_const
                    if sel_list[3]:
                        stir = 0.5* self.stir_const
                    if sel_list[4]:
                        stir = 1* self.stir_const



        print("Torque:", tor_ave)
        print("Stir  :",stir)
        self.xb = x
        self.yb = y
        self.zb = z
        return stir, tor_ave

    def move(self):
        self.env.step()
        return


env.reset()
behavior_name = list(env.behavior_specs)[0]
decision_steps, _ = env.get_steps(behavior_name)
cur_obs = decision_steps.obs[0][0,:]
dec = decider(env,1,0,150)
stir = 0
while True:
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    cur_obs = decision_steps.obs[0][0,:]
    x,y,z,xp,yp,zp,s1,s2,s3,s4,s5 = cur_obs
    #print("cur observations : ", decision_steps.obs[0][0,:])

    # Set the actions
    stir,tor = dec.decide_movement(x,y,z,xp,yp,zp,s1,s2,s3,s4,s5,stir)
    env.set_actions(behavior_name, np.array([[stir,tor,tor]]))

    # Move the simulation forward
    env.step()

env.close()
