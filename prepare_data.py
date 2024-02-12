import pickle
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import random


state_size = ....
reward_size = ....
action_size = ...

transition_size = state_size * 2 + action_size + reward_size


def get_subtraj(trajectory, number_transition, start_index):
    
    
    if start_index + number_transition - 1 >= len(trajectory):
        tp = []
        return tp
    
    subtraj = []
    subtraj = trajectory[start_index]   
    index = start_index + 1
    
    
    for ii in range(0, number_transition-1):  
        subtraj = subtraj + trajectory[index]
        index = index + 1
    
    
    if index > len(trajectory)-1:
        tp = []
        return tp
    tmp = trajectory[index][0:state_size]
    subtraj = subtraj + tmp #### s'
    
    
    
    if len(subtraj) != (transition_size + (state_size+action_size+reward+size) * (number_transition-1)) :
        print('error2', len(trajectory), ' ',len(trajectory[start_index]),' ', number_transition, ' ',start_index,' ',index,' ',len(subtraj))
    
    return subtraj




def get_triples(aim_driver :int, number_transition :int, filter_out:int, used_driver_size:int, slide_length: int, move_size:int):
    
    #cnt = 0
    
    file = open()
    tagged_traj1 = pickle.load(file)
    file.close()
    
    tuple_list = []
    for trajectory in tagged_traj1[aim_driver][1]: 
        

        

        if len(trajectory)-1 <= number_transition:  
            continue
        else:
            for i in range(len(trajectory)-1-number_transition): 
                
                
                if i>move_size:  
                    break
                
                anchor = get_subtraj(trajectory, number_transition, i)
                anchor = torch.tensor(anchor, dtype=torch.float32)

                
                positive = get_subtraj(trajectory, number_transition, i + slide_length)  
                if len(positive) == 0:
                    break
                    
                
                positive = torch.tensor(positive, dtype=torch.float32)


                ct1 = 0
                
                
                for k in range(len(tagged_traj1)):
                    
                    ct1+=1
                    if ct1>=used_driver_size:
                        break

                    if k != aim_driver:
                        
                        
                                            
                        for j in tagged_traj1[k][1]:

                            
                            random_integer = random.randint(0, 100)  
                            
                            
                            if i + number_transition + 1 < len(j) and random_integer > filter_out:
                                
                                negative = get_subtraj(j, number_transition, i)
                                negative = torch.tensor(negative, dtype=torch.float32)
                                tuple = (anchor, positive, negative)
                                
                                if len(anchor)!= transition_size + (state_size+action_size+reward+size) * (number_transition-1) or len(positive)!= transition_size + (state_size+action_size+reward+size) * (number_transition-1) or len(negative)!= transition_size + (state_size+action_size+reward+size) * (number_transition-1):
                                    print('error', len(anchor),' ',len(positive),' ',len(negative))
                                    
                                    
                                tuple_list.append(tuple)
                                
    return tuple_list                               
                                

            
        
        
def get_anchor(aim_driver :int, number_transition :int, filter_out:int, used_driver_size:int, slide_length: int, move_size:int):
    

    
   
    tagged_traj1 = pickle.load(file)
    file.close()
    
    anchor_list = []
    
    for driver_idx in range(used_driver_size):
        
        if driver_idx != aim_driver:   
            continue
        
        else:
            
        
            for trajectory in tagged_traj1[driver_idx][1]: #### 
                

                if len(trajectory)-1 <= number_transition: 
                    continue
                else:
                    for i in range(len(trajectory)-1-number_transition):
                        
                        
                        if i > move_size: 
                            break
                        
                        anchor = get_subtraj(trajectory, number_transition, i)
                        if len(anchor)!= transition_size + (state_size+action_size+reward+size) * (number_transition-1) :
                            print('error', len(anchor))
                            continue
                            
                        anchor_list.append(anchor)
                          
    return anchor_list          
        
        
        
        
        
        
def get_negative(aim_driver :int, number_transition :int, filter_out:int, used_driver_size:int, slide_length: int, move_size:int):
    
    
        
   
    tagged_traj1 = pickle.load(file)
    file.close()
    
    negative_list = []
    
    for ii in range(used_driver_size):
        
        if ii == aim_driver:
            continue
        
        for trajectory in tagged_traj1[ii][1]: 
            

            if len(trajectory)-1 <= number_transition:  
                continue
            else:
                for i in range(len(trajectory)-1-number_transition):
                    
                    
                    if i > move_size:  
                        break
                    
                    negative = get_subtraj(trajectory, number_transition, i)
                    
                    
                    if len(negative)!= transition_size + (state_size+action_size+reward+size) * (number_transition-1) :
                        print('error', len(negative))
                        continue
                        
                    negative_list.append(negative)
                          
    return negative_list  
      
