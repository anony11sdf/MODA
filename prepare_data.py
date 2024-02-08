import pickle
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import random


# def get_subtraj(trajectory, number_transition, start_index):
    
    
    
#     subtraj = []
#     subtraj = trajectory[start_index]   ## (s1, a1, r1)
#     index = start_index + 1
    
    
#     for ii in range(number_transition-1):  ### (s1, a1, r1)+(s2, a2, r2)+....
#         subtraj = subtraj + trajectory[index]
#         index = index + 1
    
#     #subtraj += trajectory[index][0:125]  #### s'
    
#     tmp = trajectory[index][0:125]
#     subtraj = subtraj + tmp #### s'
    
    
    
    
#     return subtraj




# def get_triples(aim_driver :int, number_transition :int):
    
#     cnt = 0
    
#     file = open('/home/xinbo/project1/data/reward_traj.pkl', 'rb')
#     tagged_traj = pickle.load(file)
#     file.close()
    
#     tuple_list = []
#     for trajectory in tagged_traj[aim_driver][1]: 
        
#         cnt += 1
#         #print(cnt)

#         if len(trajectory)-1 <= number_transition:  ####保证至少一个anchor一个positive
#             continue
#         else:
#             for i in range(len(trajectory)-1-number_transition): ####这个1是滑动间隔，可以考虑设为hyperparameter
                
                
                
#                 if i >2:
#                     break
                
                
                
#                 anchor = get_subtraj(trajectory, number_transition, i)
#                 positive = get_subtraj(trajectory, number_transition, i+1)
                
#                 ###get negative
                
#                 ct1=0
                
#                 for driver, trajs in tagged_traj: 
                    
#                     ct1+=1
#                     if ct1>4:
#                         break
                    
#                     if driver != aim_driver:
#                         for j in trajs: ####j 代表一个trajectory
                            
#                             if i + number_transition < len(j):
                                
#                                 negative = get_subtraj(j, number_transition, i)
#                                 tuple = (anchor, positive, negative)
                                
#                                 if len(anchor)!= 252 + 127 * (number_transition-1) or len(positive)!= 252 + 127 * (number_transition-1) or len(negative)!= 252 + 127 * (number_transition-1):
#                                     print('error', len(anchor),' ',len(positive),' ',len(negative))
                                    
                                    
#                                 tuple_list.append(tuple)
#     return tuple_list
            
        
    
    
    



def get_subtraj(trajectory, number_transition, start_index):
    
    
    if start_index + number_transition - 1 >= len(trajectory):
        #print('error1')
        tp = []
        return tp
    
    subtraj = []
    subtraj = trajectory[start_index]   ## (s1, a1, r1)
    index = start_index + 1
    
    
    for ii in range(0, number_transition-1):  ### (s1, a1, r1)+(s2, a2, r2)+....
        subtraj = subtraj + trajectory[index]
        #print('error3')
        index = index + 1
    
    
    if index > len(trajectory)-1:
        tp = []
        return tp
    tmp = trajectory[index][0:125]
    subtraj = subtraj + tmp #### s'
    
    
    
    if len(subtraj) != (252 + 127 * (number_transition-1)) :
        print('error2', len(trajectory), ' ',len(trajectory[start_index]),' ', number_transition, ' ',start_index,' ',index,' ',len(subtraj))
    
    return subtraj




def get_triples(aim_driver :int, number_transition :int, filter_out:int, used_driver_size:int, slide_length: int, move_size:int):
    
    #cnt = 0
    
    file = open('/home/xinbo/project1/data/reward_traj.pkl', 'rb')
    tagged_traj1 = pickle.load(file)
    file.close()
    
    tuple_list = []
    for trajectory in tagged_traj1[aim_driver][1]: #### dont need last trajectory
        
        #cnt += 1
        #print(cnt)
        #ct =0
        

        if len(trajectory)-1 <= number_transition:  ####保证至少一个anchor一个positive
            continue
        else:
            for i in range(len(trajectory)-1-number_transition):   ##i is start idx
                
                
                if i>move_size:  ###同一个trajectory 向右滑动的最大距离
                    break
                
                
                
                anchor = get_subtraj(trajectory, number_transition, i)
                anchor = torch.tensor(anchor, dtype=torch.float32)
                
                # if i + slide_length + number_transition > len(trajectory):
                #     break
                
                positive = get_subtraj(trajectory, number_transition, i + slide_length)  ####slide_length 这个1是滑动间隔，可以考虑设为hyperparameter
                if len(positive) == 0:
                    break
                    
                
                positive = torch.tensor(positive, dtype=torch.float32)
                ###get negative
                

                ct1 = 0 ##### control negative drivers' number
                
                
                for k in range(len(tagged_traj1)):
                    
                    ct1+=1
                    if ct1>=used_driver_size:
                        break
                    
                    
                    #print('k',k)
                    
                    
                    
                    if k != aim_driver:
                        
                        
                        
                        
                        for j in tagged_traj1[k][1]:
                            #print('len j',len(j))
                            
                            
                            random_integer = random.randint(0, 100)  #### 在当前司机下，限制选择的trajectory数量
                            
                            
                            if i + number_transition + 1 < len(j) and random_integer > filter_out:
                                
                                negative = get_subtraj(j, number_transition, i)
                                negative = torch.tensor(negative, dtype=torch.float32)
                                tuple = (anchor, positive, negative)
                                
                                if len(anchor)!= 252 + 127 * (number_transition-1) or len(positive)!= 252 + 127 * (number_transition-1) or len(negative)!= 252 + 127 * (number_transition-1):
                                    print('error', len(anchor),' ',len(positive),' ',len(negative))
                                    
                                    
                                tuple_list.append(tuple)
                                
    return tuple_list                               
                                

            
        
        
def get_anchor(aim_driver :int, number_transition :int, filter_out:int, used_driver_size:int, slide_length: int, move_size:int):
    
    #cnt = 0
    
    file = open('/home/xinbo/project1/data/reward_traj.pkl', 'rb')
    tagged_traj1 = pickle.load(file)
    file.close()
    
    anchor_list = []
    
    for driver_idx in range(used_driver_size):
        
        if driver_idx != aim_driver:   #####跳过not目标司机
            continue
        
        else:
            
        
            for trajectory in tagged_traj1[driver_idx][1]: #### 
                

                if len(trajectory)-1 <= number_transition:  ####保证至少一个sub-trajectory
                    continue
                else:
                    for i in range(len(trajectory)-1-number_transition):
                        
                        
                        if i > move_size:  ###同一个trajectory 向右滑动的最大距离
                            break
                        
                        anchor = get_subtraj(trajectory, number_transition, i)
                        #anchor = torch.tensor(anchor, dtype=torch.float32)
                        
                        #anchor_list.append(anchor)
                        if len(anchor)!= 252 + 127 * (number_transition-1) :
                            print('error', len(anchor))
                            continue
                            
                        anchor_list.append(anchor)
                          
    return anchor_list          
        
        
        
        
        
        
def get_negative(aim_driver :int, number_transition :int, filter_out:int, used_driver_size:int, slide_length: int, move_size:int):
    
    
        
    file = open('/home/xinbo/project1/data/reward_traj.pkl', 'rb')
    tagged_traj1 = pickle.load(file)
    file.close()
    
    negative_list = []
    
    for ii in range(used_driver_size):
        
        if ii == aim_driver:
            continue
        
        for trajectory in tagged_traj1[ii][1]: #### dont need last trajectory
            

            if len(trajectory)-1 <= number_transition:  ####保证至少一个anchor一个positive
                continue
            else:
                for i in range(len(trajectory)-1-number_transition):
                    
                    
                    if i > move_size:  ###同一个trajectory 向右滑动的最大距离
                        break
                    
                    negative = get_subtraj(trajectory, number_transition, i)
                    #anchor = torch.tensor(anchor, dtype=torch.float32)
                    
                    
                    if len(negative)!= 252 + 127 * (number_transition-1) :
                        print('error', len(negative))
                        continue
                        
                    negative_list.append(negative)
                          
    return negative_list  
    
    
    #cnt = 0
    
    
    
        
def get_negative_V0(aim_driver :int, number_transition :int, filter_out:int, used_driver_size:int, slide_length: int, move_size:int): ####目前广泛使用
    
    
        
    file = open('/home/xinbo/project1/data/reward_traj.pkl', 'rb')
    tagged_traj1 = pickle.load(file)
    file.close()
    
    negative_list = []
    
    
    for trajectory in tagged_traj1[aim_driver][1]: #### dont need last trajectory
        

        if len(trajectory)-1 <= number_transition:  ####保证至少一个anchor一个positive
            continue
        else:
            for i in range(len(trajectory)-1-number_transition):
                
                
                if i > move_size:  ###同一个trajectory 向右滑动的最大距离
                    break
                
                negative = get_subtraj(trajectory, number_transition, i)
                #anchor = torch.tensor(anchor, dtype=torch.float32)
                
                
                if len(negative)!= 252 + 127 * (number_transition-1) :
                    print('error', len(negative))
                    continue
                    
                negative_list.append(negative)
                          
    return negative_list  
    
    
    
    
    
    # file = open('/home/xinbo/project1/data/reward_traj.pkl', 'rb')
    # tagged_traj1 = pickle.load(file)
    # file.close()
    
    # negative_list = []
    # for trajectory in tagged_traj1[aim_driver][1]: #### dont need last trajectory
        
    #     if len(trajectory)-1 <= number_transition:  ####保证至少一个anchor一个positive
    #         continue
    #     else:
    #         for i in range(len(trajectory)-1-number_transition):
                
                
    #             if i>move_size:  ###同一个trajectory 向右滑动的最大距离
    #                 break
                              
    #             #anchor = get_subtraj(trajectory, number_transition, i)
    #             #anchor = torch.tensor(anchor, dtype=torch.float32)

    #             ct1 = 0 ##### control negative drivers' number
                
                
    #             for k in range(len(tagged_traj1)):
                    
    #                 ct1+=1
    #                 if ct1>=used_driver_size:
    #                     break
 
                    
    #                 if k != aim_driver:
                        
                         
    #                     for j in tagged_traj1[k][1]:
    #                         #print('len j',len(j))
                            
                            
    #                         random_integer = random.randint(0, 100)  #### 在当前司机下，限制选择的trajectory数量
                            
                            
    #                         if i + number_transition + 1 < len(j) and random_integer > filter_out:
                                
    #                             negative = get_subtraj(j, number_transition, i)
    #                             #negative = torch.tensor(negative, dtype=torch.float32)
                                
                                
    #                             if len(negative)!= 252 + 127 * (number_transition-1):
    #                                 print('error',len(negative))
                                    
                                    
    #                             negative_list.append(negative)
                                
    # return negative_list         
        
        
        
        
        
        
        
            
            
def get_triples_V2(aim_driver :int, number_transition :int, filter_out:int, used_driver_size:int, slide_length: int, move_size:int): ########更节省空间，或许用不到...
    
    
    
    
    
    
    
    
    
    
    
    
    
    #cnt = 0
    
    file = open('/home/xinbo/project1/data/reward_traj.pkl', 'rb')
    tagged_traj1 = pickle.load(file)
    file.close()
    
    tuple_list = []
    for trajectory in tagged_traj1[aim_driver][1]: #### dont need last trajectory
        
        #cnt += 1
        #print(cnt)
        #ct =0
        

        if len(trajectory)-1 <= number_transition:  ####保证至少一个anchor一个positive
            continue
        else:
            for i in range(len(trajectory)-1-number_transition):
                
                
                if i>move_size:  ###同一个trajectory 向右滑动的最大距离
                    break
                
                
                
                anchor = get_subtraj(trajectory, number_transition, i)
                anchor = torch.tensor(anchor, dtype=torch.float32)
                
                # if i + slide_length + number_transition > len(trajectory):
                #     break
                
                positive = get_subtraj(trajectory, number_transition, i + slide_length)  ####slide_length 这个1是滑动间隔，可以考虑设为hyperparameter
                if len(positive) == 0:
                    break
                    
                
                positive = torch.tensor(positive, dtype=torch.float32)
                ###get negative
                

                ct1 = 0 ##### control negative drivers' number
                
                
                for k in range(len(tagged_traj1)):
                    
                    ct1+=1
                    if ct1>=used_driver_size:
                        break
                    
                    
                    #print('k',k)
                    
                    
                    
                    if k != aim_driver:
                        
                        
                        
                        
                        for j in tagged_traj1[k][1]:
                            #print('len j',len(j))
                            
                            
                            random_integer = random.randint(0, 100)  #### 在当前司机下，限制选择的trajectory数量
                            
                            
                            if i + number_transition + 1 < len(j) and random_integer > filter_out:
                                
                                negative = get_subtraj(j, number_transition, i)
                                negative = torch.tensor(negative, dtype=torch.float32)
                                tuple = (anchor, positive, negative)
                                
                                if len(anchor)!= 252 + 127 * (number_transition-1) or len(positive)!= 252 + 127 * (number_transition-1) or len(negative)!= 252 + 127 * (number_transition-1):
                                    print('error', len(anchor),' ',len(positive),' ',len(negative))
                                    
                                    
                                tuple_list.append(tuple)
                                
    return tuple_list     