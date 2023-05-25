####This File consists of all functions related to general utilities####################
####Maintainer: Omey Mohan Manyar, Email ID: manyar@usc.edu#############################

import numpy as np
import copy
from string import ascii_lowercase
import csv
import sys
from queue import PriorityQueue
import re
import json
import matplotlib.font_manager as font_manager
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance

############ General Utilities #######################

####################################################################################
#Utility function to check if two trajectories are similar or not by taking the sum of their features
def check_if_traj_similar(graph, traj_1_list, traj_2_list, similarity_threshold = 1):

    similarity_flag = False

    previous_traj1_state = traj_1_list[0]
    previous_traj2_state = traj_2_list[0]

    current_traj1_state = traj_1_list[1]
    no_of_features = len(graph[str(previous_traj1_state)].get_child_features(current_traj1_state))

    sum_feature_vector_traj1 = np.zeros(no_of_features)
    sum_feature_vector_traj2 = np.zeros(no_of_features)
    
    
    for i in range(1,len(traj_1_list)):
        current_traj1_state = traj_1_list[i]
        current_traj2_state = traj_2_list[i]

        current_traj1_state_feature_vec = graph[str(previous_traj1_state)].get_child_features(current_traj1_state)
        current_traj2_state_feature_vec = graph[str(previous_traj2_state)].get_child_features(current_traj2_state)
        
        sum_feature_vector_traj1 += current_traj1_state_feature_vec
        sum_feature_vector_traj2 += current_traj2_state_feature_vec

        previous_traj1_state = current_traj1_state
        previous_traj2_state = current_traj2_state


    similarity_vector = np.zeros(no_of_features, dtype=bool)
  
    
    for i in range(len(sum_feature_vector_traj1)):
        if(sum_feature_vector_traj1[i] == sum_feature_vector_traj2[i]):
            similarity_vector[i] = True


    if(False not in similarity_vector):
        similarity_flag = True

    return similarity_flag

#######################################################################################################
def check_if_traj_similar_hamming(graph, traj_1_list, traj_2_list, similarity_threshold = 0.01):

    similarity_flag = False

    previous_traj1_state = traj_1_list[0]
    previous_traj2_state = traj_2_list[0]

    current_traj1_state = traj_1_list[1]
    no_of_features = len(graph[str(previous_traj1_state)].get_child_features(current_traj1_state))

    sum_feature_vector_traj1 = np.zeros(no_of_features)
    sum_feature_vector_traj2 = np.zeros(no_of_features)
    
    for i in range(1,len(traj_1_list)):
        current_traj1_state = traj_1_list[i]
        current_traj2_state = traj_2_list[i]

        current_traj1_state_feature_vec = graph[str(previous_traj1_state)].get_child_features(current_traj1_state)
        current_traj2_state_feature_vec = graph[str(previous_traj2_state)].get_child_features(current_traj2_state)
        
        sum_feature_vector_traj1 += current_traj1_state_feature_vec
        sum_feature_vector_traj2 += current_traj2_state_feature_vec

        previous_traj1_state = current_traj1_state
        previous_traj2_state = current_traj2_state


  
    hamming_distance = compute_hamming_distance(sum_feature_vector_traj1, sum_feature_vector_traj2)
    
    if(hamming_distance < similarity_threshold):
        similarity_flag = True
    

    return similarity_flag

#########################################################

#####This function returns the cosine similarity between two feature vectors
def check_if_traj_similar_cosine(graph, traj_1_list, traj_2_list, similarity_threshold = 0.2):
    similarity_flag = False

    previous_traj1_state = traj_1_list[0]
    previous_traj2_state = traj_2_list[0]

    current_traj1_state = traj_1_list[1]
    no_of_features = len(graph[str(previous_traj1_state)].get_child_features(current_traj1_state))

    sum_feature_vector_traj1 = np.zeros(no_of_features)
    sum_feature_vector_traj2 = np.zeros(no_of_features)
    
    for i in range(1,len(traj_1_list)):
        current_traj1_state = traj_1_list[i]
        current_traj2_state = traj_2_list[i]

        current_traj1_state_feature_vec = graph[str(previous_traj1_state)].get_child_features(current_traj1_state)
        current_traj2_state_feature_vec = graph[str(previous_traj2_state)].get_child_features(current_traj2_state)
        
        sum_feature_vector_traj1 += current_traj1_state_feature_vec
        sum_feature_vector_traj2 += current_traj2_state_feature_vec

        previous_traj1_state = current_traj1_state
        previous_traj2_state = current_traj2_state


  
    cosine_similarity = compute_cosine_similarity(sum_feature_vector_traj1, sum_feature_vector_traj2)

    cosine_distance = 1 - cosine_similarity
    
    if(cosine_distance <= similarity_threshold):
        similarity_flag = True
    

    return similarity_flag



## Returns the corresponding alphabet for a given numeric positions
## i.e. num_value = 1 returns a while num_value = 26 return z
def num2alpha(num_value):
    num2alpha_dict = dict(zip(range(1, 27), ascii_lowercase))
    return num2alpha_dict[num_value]

## Returns the corresponding numeric positions for a given alphabet
## i.e. alphabet = z returns 26 (Alphabets have to be small)
def alpha2num(alphabet):
    position = ord(alphabet) - 96
    return position


#####read input csv##########
def read_csv(filename):
    rows = []
    fields = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        fields = next(csvreader)

        for row in csvreader:
            rows.append(row)
        

    return [fields, rows]

#Converts one hot vector to path id list
def convert_onehot_to_region_id(path):
    
    no_of_regions = len(path[0])
    previous_region = np.zeros(no_of_regions, dtype=int)
    previous_region = path[0]

    region_id_path = []

    
    for i in range(1,len(path)):
        current_region =  path[i]
        action_vector = np.zeros(no_of_regions)
        action_vector = current_region - previous_region
        
        action_id = np.where(action_vector == 1)[0]
        action_id = action_id[0] + 1
        
        region_id_path = np.append(region_id_path,action_id)
        previous_region = path[i]
        
    
    region_id_path = region_id_path.astype(int)
    
    return region_id_path

#######Converts a Region ID Path to One Hot Encoding Path List #################
def convert_regionid_to_onehotvector(region_id_path):
    no_of_regions = len(region_id_path)
    one_hot_vector = np.zeros(no_of_regions)
    
    previous_region = np.zeros(no_of_regions, dtype=int)
    one_hot_vector = np.array([previous_region])
    
    for i in range(len(region_id_path)):
        current_region = np.zeros(no_of_regions, dtype=int)
        current_region[region_id_path[i]-1] = 1
        current_region = previous_region + current_region
        current_region = current_region.astype(int)
        one_hot_vector = np.append(one_hot_vector, current_region)
        previous_region = current_region 
    

    one_hot_vector = np.reshape(one_hot_vector, [no_of_regions+1, no_of_regions])
    one_hot_vector = one_hot_vector.astype(int)
    
    return one_hot_vector

######Converts a String to a pathof integers#################
def convert_string_to_path(current_path_str, no_of_regions):
    
    internal_loop_count = 2*no_of_regions + 1
    number_of_paths = int(len(current_path_str)/(internal_loop_count+2))
    
    if(number_of_paths == 0):
        number_of_paths = 1
    
    all_available_paths = re.findall(r'\d+', current_path_str)
    all_available_paths = np.array(all_available_paths)
    all_available_paths = all_available_paths.astype(int)
    all_available_paths = np.reshape(all_available_paths, [number_of_paths, no_of_regions])
    
    
    return all_available_paths



################################################################################
#######Function to convert an alphabet based region path to a region id path####
def convert_alpha_path_to_num_path(alpha_path):

    no_of_regions = len(alpha_path)

    num_path = np.zeros(no_of_regions, dtype=int)

    for i in range(no_of_regions):
        current_region_id = alpha2num(alpha_path[i])
        num_path[i] = int(current_region_id)

    num_path = num_path.astype(int)

    return num_path


################################################################################
#######Function to convert a region id number based path to an alphabet based path####
def convert_num_path_to_alpha_path(num_path):

    no_of_regions = len(num_path)

    alpha_path = ['a'] * no_of_regions

    for i in range(no_of_regions):
        current_region_alpha_id = num2alpha(num_path[i])
        alpha_path[i] = current_region_alpha_id

    
    return alpha_path

####################################################################################
############ Graph based Shortest Path Computation Utilities #######################
####################################################################################
###This should be used after the dijstra's algorithm below###
### Given the start node, target node and previous nodes dictionary###
### This gives the shortest path from start to target########
def get_dijstra_result(start_node, target_node, previous_nodes):
        path = []
        node = target_node
        count = 0
        region_id = 0
        solution_path = []
        path_count = 0
        no_of_regions = len(start_node)
        
        while str(node) != str(start_node):
            
            if(count!=0):
                
                previous_node = previous_node.astype(int)
                
                region_id = np.subtract(previous_node,node)
                region_id = np.where(region_id == 1)[0] + 1
                region_id = region_id[0].astype(int)
                solution_path = np.append(solution_path,region_id)
            
            path = np.append(path,node)
            path_count += 1
            previous_node = node
            
            try:
                node = previous_nodes[str(node)]
            except:
                return [None, None, False]
                
            count+=1
    
        # Add the start node manually
        region_id = previous_node
        region_id = np.where(region_id == 1)[0] + 1
        
        region_id = region_id[0].astype(int)
        
        solution_path = np.append(solution_path,region_id)
        solution_path = solution_path.astype(int)
        
        path = np.append(path,start_node)
        path_count += 1
        
        path = np.reshape(path,[path_count, no_of_regions])
        path = path.astype(int)
      
        shortest_path_list = np.flip(path, axis=0)
        shortest_path_id_list = np.flip(solution_path)
        
        
        return [shortest_path_id_list, shortest_path_list, [True]]


####Given a graph and the start node this function evaluates the shortest path for the start node to each node####
#######This is the Implementation of the Dijstra's Algorithm####################
def shortest_path_dijstra(graph,start_node):

    shortest_path = {}
    previous_nodes = {}
    dist = dict.fromkeys(graph.keys(), sys.maxsize)
    visited_queue = dict.fromkeys(graph.keys(), False)

    no_of_regions = len(start_node)
    
    ##Initializing the distance vector
    dist[str(start_node)] = 0
    shortest_path[str(start_node)] = 0
    visited_priority_queue = PriorityQueue()
    visited_priority_queue.put([0,str(start_node)])

    while(not visited_priority_queue.empty()):
        
        node = visited_priority_queue.get()[1]
    
        x = np.zeros(no_of_regions, dtype=int)
        
        for i in range (len(node)):
            if(i%2!=0):
                x[int(i/2)] = int(node[i])
        
        
        visited_queue[str(x)] = True
        
    
        children_of_x = copy.deepcopy(list(graph[str(x)].get_children_ids()))
        
        for i in range (len(children_of_x)):
            edge_weight = 0.0
            edge_weight = graph[str(x)].get_edge_weight(children_of_x[i])
            
            cost = edge_weight + dist[str(x)]
        
            if(visited_queue[str(children_of_x[i])] == False and dist[str(children_of_x[i])] > cost):
                
                dist[str(children_of_x[i])] = dist[str(x)] + graph[str(x)].get_edge_weight(children_of_x[i])
                shortest_path[str(children_of_x[i])] = dist[str(children_of_x[i])]
                previous_nodes[str(children_of_x[i])] = x
                visited_priority_queue.put([dist[str(children_of_x[i])],children_of_x[i]])
                
    
    return [shortest_path, previous_nodes]


######This is the Implementation of the Yen's Shortest Path Algorithm
def k_Shortest_Paths(k, graph, source_node, goal_node, similarity_check_path):
        
        
        nth_shortest_path_queue = PriorityQueue()
        n_path_list_id = []
        n_path_list = []
        
        no_of_regions = len(source_node)
        
        similarity_threshold = 0.000000001
        [shortest_path_to_each_node, previous_nodes_dict] = shortest_path_dijstra(graph, source_node)
        [shortest_path_id_list,shortest_path_list, bool_flag] =  get_dijstra_result(source_node, goal_node, previous_nodes_dict)

        n_path_list_id = np.append(n_path_list_id, [shortest_path_id_list]) 
        n_path_list_id = np.reshape(n_path_list_id, [1,len(n_path_list_id) ])
        n_path_list_id = n_path_list_id.astype(int) 

        n_path_list = np.append(n_path_list, [shortest_path_list])
        
        n_path_list = np.reshape(n_path_list, [1,no_of_regions+1,no_of_regions])
        n_path_list = n_path_list.astype(int)
    
        true_shortest_path_id = copy.deepcopy(shortest_path_id_list)
        true_shortest_path_list =  copy.deepcopy(shortest_path_list)

        sampled_traj_count = 1
        similar_traj_ids = []
        loop_count = 1

        similarity_check_path_list = convert_regionid_to_onehotvector(similarity_check_path)
        
        copy_of_graph = copy.deepcopy(graph)

        while(sampled_traj_count != k+1):
            [n_traj, n_path_row, n_path_col] = np.shape(n_path_list)
            
            current_solution_path = n_path_list[n_traj-1]
            
            for j in range(0,len(current_solution_path)-3):
                
                spur_node = current_solution_path[j]
                spur_node = spur_node.astype(int)
                rootPath = current_solution_path[0:j+1]
                
                for path in n_path_list:
                    
                    if(str(rootPath) == str(path[0:j+1])):
                        copy_of_graph[str(path[j])].remove_child(path[j+1])
                
                for nodes in rootPath:
                    if(str(nodes) != str(spur_node)):
                        copy_of_graph.pop(str(nodes), None)

                [temp_shortest_path_to_each_node, temp_previous_nodes] = shortest_path_dijstra(copy_of_graph,spur_node)
                
                ##Dijstra Result: Shortest Path ID, Shortest Path List, Bool Flag
                dijstra_result = get_dijstra_result(spur_node, goal_node, temp_previous_nodes)
                
                if(not dijstra_result[2]):
                    copy_of_graph = copy.deepcopy(graph) 
                    break      
                
                ###SpurPath is the one hot ID vector
                spurPath = dijstra_result[1]
                
                total_path = np.append(rootPath, spurPath[1:])
                total_path = np.reshape(total_path, [no_of_regions+1, no_of_regions])
                
                region_id_path = convert_onehot_to_region_id(total_path)
                
                if(str(region_id_path) not in str(n_path_list_id)):
                    
                    cost = getPathCost(graph, region_id_path)    
                    nth_shortest_path_queue.put([cost, str(region_id_path)])
                
                copy_of_graph = copy.deepcopy(graph)
                
            
            if(nth_shortest_path_queue.empty()):
                print("Exited the for loop due to length being zero")
                break
          
            lowest_cost_path_str = copy.deepcopy(nth_shortest_path_queue.get())
            current_nth_shortest_path_id = convert_string_to_path(lowest_cost_path_str[1], no_of_regions)
            
            for path in current_nth_shortest_path_id:
                current_path_list = convert_regionid_to_onehotvector(path)
                
                if(str(path) not in str(n_path_list_id)):
                    
                    [row_count, col_count] = np.shape(n_path_list_id)
                    n_path_list_id = np.append(n_path_list_id, path)
                    n_path_list_id = np.reshape(n_path_list_id, [row_count+1,col_count])
                    
                    
                    [row,col] = np.shape(current_path_list)
                    current_path_list = current_path_list.astype(int)
                    current_path_list = np.reshape(current_path_list,[no_of_regions+1,no_of_regions])
                    
                    [l,r,c] = np.shape(n_path_list)
                    n_path_list = np.append(n_path_list, current_path_list) 
                    n_path_list = np.reshape(n_path_list, [l+1,row, col])
                    
                    if(check_if_traj_similar_cosine(graph, similarity_check_path_list, current_path_list, similarity_threshold)):
                        similar_traj_ids = np.append(similar_traj_ids,int(row_count+1))
                    else:
                        sampled_traj_count+=1

            loop_count+=1
            
        similar_traj_ids = np.asarray(similar_traj_ids)
        similar_traj_ids = similar_traj_ids.astype(int)
        
        n_paths = np.asarray(n_path_list_id)
        n_paths = n_paths.astype(int)
        n_paths = np.delete(n_paths, similar_traj_ids, axis=0)
        
        if(str(similarity_check_path) in str(n_paths)):
           
            index_to_remove = []
            for x in range(len(n_paths)):
                if(np.array_equal(n_paths[x], similarity_check_path)):
                    index_to_remove.append(x)
            
        
            n_paths = np.delete(n_paths, index_to_remove, axis = 0)
        
        else:
            n_paths = n_paths[:-1]          #Removing the last element
        
    

        return n_paths


######This is the Implementation of the Yen's Shortest Path Algorithm
def k_similar_paths(mold):
        
        
        nth_shortest_path_queue = PriorityQueue()
        n_path_list_id = []
        n_path_list = []

        graph = copy.deepcopy(mold.vert_dict)
        source_node = mold.source_node
        goal_node = mold.goal_node
        similarity_check_path = mold.user_preferred_path_num
        
        no_of_regions = len(source_node)
        # similarity_threshold = (1/no_of_regions) + 0.01
        similarity_threshold = 0.0001
        [shortest_path_to_each_node, previous_nodes_dict] = shortest_path_dijstra(graph, source_node)
        [shortest_path_id_list,shortest_path_list, bool_flag] =  get_dijstra_result(source_node, goal_node, previous_nodes_dict)

        n_path_list_id = np.append(n_path_list_id, [shortest_path_id_list]) 
        n_path_list_id = np.reshape(n_path_list_id, [1,len(n_path_list_id) ])
        n_path_list_id = n_path_list_id.astype(int) 

        n_path_list = np.append(n_path_list, [shortest_path_list])
        n_path_list = np.reshape(n_path_list, [1,no_of_regions+1,no_of_regions])
        n_path_list = n_path_list.astype(int)
    
        true_shortest_path_id = copy.deepcopy(shortest_path_id_list)
        true_shortest_path_list =  copy.deepcopy(shortest_path_list)

        sampled_traj_count = 1
        similar_traj_ids = []
        loop_count = 1

        similarity_check_path_list = convert_regionid_to_onehotvector(similarity_check_path)
        
        copy_of_graph = copy.deepcopy(graph)
        current_traj_cost = getPathCost(graph, similarity_check_path)
        
        sampled_traj_cost = 0

        while(sampled_traj_cost <= current_traj_cost):
            [n_traj, n_path_row, n_path_col] = np.shape(n_path_list)
            
            current_solution_path = n_path_list[n_traj-1]
            
            for j in range(0,len(current_solution_path)-3):
                
                spur_node = current_solution_path[j]
                spur_node = spur_node.astype(int)
                rootPath = current_solution_path[0:j+1]
                
                for path in n_path_list:
                    
                    if(str(rootPath) == str(path[0:j+1])):
                        copy_of_graph[str(path[j])].remove_child(path[j+1])
                
                for nodes in rootPath:
                    if(str(nodes) != str(spur_node)):
                        copy_of_graph.pop(str(nodes), None)

                [temp_shortest_path_to_each_node, temp_previous_nodes] = shortest_path_dijstra(copy_of_graph,spur_node)
                
                ##Dijstra Result: Shortest Path ID, Shortest Path List, Bool Flag
                dijstra_result = get_dijstra_result(spur_node, goal_node, temp_previous_nodes)
                
                if(not dijstra_result[2]):
                    copy_of_graph = copy.deepcopy(graph) 
                    break      
                
                ###SpurPath is the one hot ID vector
                spurPath = dijstra_result[1]
                
                total_path = np.append(rootPath, spurPath[1:])
                total_path = np.reshape(total_path, [no_of_regions+1, no_of_regions])
                
                region_id_path = convert_onehot_to_region_id(total_path)
                if(str(region_id_path) not in str(n_path_list_id)):
                    
                    cost = getPathCost(graph, region_id_path)    
                    nth_shortest_path_queue.put([cost, str(region_id_path)])
                
                copy_of_graph = copy.deepcopy(graph)
                
            
            if(nth_shortest_path_queue.empty()):
                print("Exited the for loop due to length being zero")
                break
          
            lowest_cost_path_str = copy.deepcopy(nth_shortest_path_queue.get())
            current_nth_shortest_path_id = convert_string_to_path(lowest_cost_path_str[1], no_of_regions)
            
            for path in current_nth_shortest_path_id:
                current_path_list = convert_regionid_to_onehotvector(path)
                
                if(str(path) not in str(n_path_list_id)):
                    [row_count, col_count] = np.shape(n_path_list_id)
                    n_path_list_id = np.append(n_path_list_id, path)
                    n_path_list_id = np.reshape(n_path_list_id, [row_count+1,col_count])
                    
                    
                    [row,col] = np.shape(current_path_list)
                    current_path_list = current_path_list.astype(int)
                    current_path_list = np.reshape(current_path_list,[no_of_regions+1,no_of_regions])
                    
                    [l,r,c] = np.shape(n_path_list)
                    n_path_list = np.append(n_path_list, current_path_list) 
                    n_path_list = np.reshape(n_path_list, [l+1,row, col])
                    sampled_traj_cost = getPathCost(graph, path)
                    sampled_traj_count+=1
                    similar_traj_ids = np.append(similar_traj_ids,int(row_count+1))
                        

            loop_count+=1
            
            
            
        n_paths = np.asarray(n_path_list_id)
        n_paths = n_paths.astype(int)
        
        if(str(similarity_check_path) in str(n_paths)):
           
            index_to_remove = []
            for x in range(len(n_paths)):
                if(np.array_equal(n_paths[x], similarity_check_path)):
                    index_to_remove.append(x)
            
            
            n_paths = np.delete(n_paths, index_to_remove, axis = 0)
            
            
        n_paths = n_paths[:-1]          #Removing the last element
        
        return n_paths

#####Given a path and a graph this returns the cost of the path#####
def getPathCost(graph, path):
    
    no_of_regions = len(path)
    previous_region = np.zeros(no_of_regions, dtype=int)
    
    cost = 0.0
    cost_vector = []
    
    for i in range(len(path)):
        current_region = np.zeros(no_of_regions, dtype=int)
        current_region[path[i]-1] = 1
        current_region = previous_region + current_region
        current_region = current_region.astype(int)
        cost += graph[str(previous_region)].get_edge_weight(current_region)
    
        cost_vector = np.append(cost_vector,cost)
        previous_region = current_region
    
    
    return cost

####################################################################################
########################## Max Margin Training Utilities ###########################
####################################################################################

###No of trajectories the total number of non similar trajectories to sample##
### Graph obj is the object of the class Graph in datastructures #############
def sample_non_similar_trajectories(no_of_trajectories, graph_obj):
    
    #####The array to hold the sampled trajectories#####
    sampled_trajectories = []

    no_of_regions = graph_obj.no_of_regions
    # similarity_threshold = (1/no_of_regions) + 0.01
    similarity_threshold = 0.005
    ###The counter for number of trajectories sampled
    sampled_trajectory_count = 0

    ##This dictionary helps in evaluating if a certain trajectory has already been sampled or not
    sampled_trajectory_dict = {}
    sampled_trajectory_feature_dict = {}

    ###Initial Random trajectory for sampling
    initial_traj = np.arange(1, no_of_regions+1)

    ###Getting the user preferred path
    user_preferred_path_list = convert_regionid_to_onehotvector(graph_obj.user_preferred_path_num)

    ###Weight of the sampled trajectories
    sampled_trajectory_weights = np.zeros(no_of_trajectories)

    ##User preferred trajectory features
    user_preferred_path_features = getPathFeatures(graph_obj.vert_dict, graph_obj.user_preferred_path_num)
    user_preferred_path_features_vec_sum = np.sum(user_preferred_path_features, axis=0)
    user_preferred_path_features_vec_sum = user_preferred_path_features_vec_sum.astype(int)

    ###Generating the trajectories
    while(sampled_trajectory_count != no_of_trajectories):

        current_sampled_trajectory = np.random.permutation(initial_traj)
        
        ###Check if the current sample has already been sampled before
        if(str(current_sampled_trajectory) in sampled_trajectory_dict.keys()):
            continue
        else:
            sampled_trajectory_dict[str(current_sampled_trajectory)] = True

        ###Now check if the sampled trajectory is similar with any trajectory that is already sampled
        ###Currently we check only against one trajectory which is the user preferred trajectory
        current_sampled_trajectory_list = convert_regionid_to_onehotvector(current_sampled_trajectory)
        if(check_if_traj_similar_cosine(graph_obj.vert_dict, user_preferred_path_list, current_sampled_trajectory_list, similarity_threshold)):
            sampled_trajectory_dict[str(current_sampled_trajectory)] = True
            continue
        else:
            sampled_trajectory_dict[str(current_sampled_trajectory)] = True
        
        ####Sample trajectories that are not similar to any other previously sampled trajectories
        current_traj_feature_vec = getPathFeatures(graph_obj.vert_dict, current_sampled_trajectory)
        current_traj_feature_vec_sum = np.sum(current_traj_feature_vec, axis=0)
        
        if(str(current_traj_feature_vec_sum) in sampled_trajectory_feature_dict.keys()):
            # print('Similar featured trajecotry found!')
            continue
        else:
            sampled_trajectory_feature_dict[str(current_traj_feature_vec_sum)] = True

        ####When both the above conditions are met, we get a unique non similar path##
        ### Adding this to the sampled trajectories
        
        if(len(sampled_trajectories) == 0):
            sampled_trajectories = [current_sampled_trajectory]
        else: 
            sampled_trajectories = np.append(sampled_trajectories, [current_sampled_trajectory], axis=0)
        
        
        sampled_trajectory_weights[sampled_trajectory_count] = len(user_preferred_path_features_vec_sum) * 10*(1-compute_cosine_similarity(user_preferred_path_features_vec_sum, current_traj_feature_vec_sum))
        sampled_trajectory_dict[str(current_sampled_trajectory)] = True

        sampled_trajectory_count += 1

    
    return [sampled_trajectories, sampled_trajectory_weights]


def sample_trajectories_random(no_of_trajectories, graph_obj):
    
    #####The array to hold the sampled trajectories#####
    sampled_trajectories = []

    no_of_regions = graph_obj.no_of_regions
    similarity_threshold = (1/no_of_regions) + 0.01

    ###The counter for number of trajectories sampled
    sampled_trajectory_count = 0

    ##This dictionary helps in evaluating if a certain trajectory has already been sampled or not
    sampled_trajectory_dict = {}
    sampled_trajectory_feature_dict = {}

    ###Initial Random trajectory for sampling
    initial_traj = np.arange(1, no_of_regions+1)

    ###Getting the user preferred path
    user_preferred_path_list = convert_regionid_to_onehotvector(graph_obj.user_preferred_path_num)

    ###Weight of the sampled trajectories
    sampled_trajectory_weights = np.zeros(no_of_trajectories)

    ##User preferred trajectory features
    user_preferred_path_features = getPathFeatures(graph_obj.vert_dict, graph_obj.user_preferred_path_num)
    user_preferred_path_features_vec_sum = np.sum(user_preferred_path_features, axis=0)
    user_preferred_path_features_vec_sum = user_preferred_path_features_vec_sum.astype(int)

    ###Generating the trajectories
    while(sampled_trajectory_count != no_of_trajectories):

        current_sampled_trajectory = np.random.permutation(initial_traj)
        
        ###Check if the current sample has already been sampled before
        if(str(current_sampled_trajectory) in sampled_trajectory_dict.keys()):
            continue
        else:
            sampled_trajectory_dict[str(current_sampled_trajectory)] = True

        current_traj_feature_vec = getPathFeatures(graph_obj.vert_dict, current_sampled_trajectory)
        current_traj_feature_vec_sum = np.sum(current_traj_feature_vec, axis=0)
        
        if(len(sampled_trajectories) == 0):
            sampled_trajectories = [current_sampled_trajectory]
        else: 
            sampled_trajectories = np.append(sampled_trajectories, [current_sampled_trajectory], axis=0)
        

        sampled_trajectory_weights[sampled_trajectory_count] = len(user_preferred_path_features_vec_sum) * 10*(1-compute_cosine_similarity(user_preferred_path_features_vec_sum, current_traj_feature_vec_sum))
        sampled_trajectory_dict[str(current_sampled_trajectory)] = True

        sampled_trajectory_count += 1


    return [sampled_trajectories, sampled_trajectory_weights]


#####This function returns the hamming distance between two feature vectors
def compute_hamming_distance(preferred_path_features, sampled_path_features):
    hamming_distance = 0.0
    hamming_distance = distance.hamming(preferred_path_features, sampled_path_features)
    return hamming_distance

#####This function returns the cosine similarity between two feature vectors
def compute_cosine_similarity(preferred_path_features, sampled_path_features):
    cosine_similarity = 0.0
    cosine_similarity = np.dot(preferred_path_features, sampled_path_features)/(np.linalg.norm(preferred_path_features)*np.linalg.norm(sampled_path_features))

    return cosine_similarity



###Normalize an array
# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr
    
#####This function returns the feature vector for a given trajectory####
def getPathFeatures(graph, path):

    traj_feature_vec = []
    no_of_regions = len(path)

    previous_state = np.zeros(no_of_regions, dtype=int)

    for i in range(no_of_regions):
        current_state = copy.deepcopy(previous_state)
        current_state[path[i]-1] = 1

        state_feature_vector = graph[str(previous_state)].get_child_features(current_state)

        if(i == 0):
            traj_feature_vec = [state_feature_vector]
        else:
            traj_feature_vec = np.append(traj_feature_vec, [state_feature_vector], axis=0)
        
        previous_state = copy.deepcopy(current_state)

    
    return traj_feature_vec

##Function to save the training data
def save_training_data(information_dictionary, mold_number, iteration_number):

    fields = information_dictionary.keys()

    filename = "Mold_"+ str(mold_number)+"_results_" + str(iteration_number) + ".txt"
    f = open(filename, 'w')

    f.write(str(information_dictionary))

    f.close()

    return


###Getting all possible paths in a graph###
def get_all_possible_paths(start, goal, visited, path, graph, shortest_path_queue):
    # Mark the current node as visited and store in path
    visited[str(start)]= True
    path.append(start)

    # If current vertex is same as destination, then print
    
    if str(start) == str(goal):
        # Remove current vertex from path[] and mark it as unvisited
        # visited[str(goal)]= False
        current_path = np.asarray(path)
        
        current_path = convert_onehot_to_region_id(path)
        cost_of_path = getPathCost(graph.vert_dict, current_path)
        
        shortest_path_queue.put([cost_of_path, str(current_path)])
        
    else:
        # If current vertex is not destination
        # Recur for all the vertices adjacent to this vertex
        current_node_children = graph.vert_dict[str(start)].get_children_ids()
      
        for child in current_node_children:
            
            if visited[str(child)] == False:
                child_state = convert_string_to_path(child, graph.no_of_regions)
                get_all_possible_paths(child_state[0], goal, visited, path, graph, shortest_path_queue)
                
    # Remove current vertex from path[] and mark it as unvisited
    path.pop()
    visited[str(start)] = False

    


###Utility function to generate paths by brute force####
def find_k_shortest_path_brute(k, graph):

    k_shortest_paths = []

    shortest_path_queue = PriorityQueue()

    no_of_regions = graph.no_of_regions
    
    source_node = graph.source_node
    goal_node = graph.goal_node

    visited = dict.fromkeys(graph.vert_dict.keys(), False)

    path = []

    print('Starting the Sampling')
    get_all_possible_paths(source_node, goal_node, visited, path, graph, shortest_path_queue)

    
    return k_shortest_paths


###Function to read the feature interaction csv
def read_json(filename):

    f = open(filename)

    data = json.load(f)

    
    return data

###Parsing the data for feature interaction values
def initialize_feature_interaction_data(interaction_data, top_level_data_keys, corresponding_key_data):

    feature_id_vec = []
    l2_feature_interactions = []
    feature_threshold = []

    for i in range(len(top_level_data_keys)):
        current_top_level_data = interaction_data[top_level_data_keys[i]]
        current_keys = corresponding_key_data[i]
        if(i==1):
            current_keys = feature_id_vec
        for j in range(len(current_top_level_data)):
           
            for z in range(len(current_keys)):
                if(i==1):
                    current_data = current_top_level_data[j+z][current_keys[z]]
                else:
                    current_data = current_top_level_data[j][current_keys[z]]

                
                if(i == 0):
                    feature_id_vec = np.array(current_data)
                
                elif(i==1):  
                    feature_threshold.append(current_data)
                elif(i == 2):
                    if(current_keys[z] == 'interacting_feature_1' or current_keys[z] == 'interacting_feature_2'):
                
                        l2_feature_interactions.append(current_data)

            if(i==1):
                break       
    
    l2_feature_interactions = np.reshape(l2_feature_interactions,[int(len(l2_feature_interactions)/2),2])
    
    return [feature_id_vec, l2_feature_interactions, feature_threshold]



#####Function to save feature values###########

def save_feature_values(filename, input_data_dictionary):

    fields = []
    fields.append('demo_no')
    data_matrix = []
    
    traj_id_vec = input_data_dictionary['traj_id']
    data_matrix.append(np.arange(1,len(traj_id_vec)+1, dtype=int))

    for key in input_data_dictionary.keys():
        fields.append(key)
        current_field_value = input_data_dictionary[key]
        data_matrix.append(current_field_value)
    

    data_matrix = np.transpose(data_matrix)

    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(data_matrix)

    return

##Key level 1 is variation in feature values e.g. HH, HL, MM, etc
##Key level 2 is the feature values
def save_feature_interaction(filename, interaction_wise_count_dict, interaction_wise_feature_variation_dict, interacting_features, key_level_1, key_level_2):

    fields = []
    
    fields.append('interacting_features')

    fields.append('interaction_type')
    fields.append('total_count')

    for key in key_level_2:
        fields.append(key)


    rows = []
    for key_1 in range(len(key_level_1)):
        current_row = []
        current_row.append(interacting_features)
        if(key_level_1[key_1] not in interaction_wise_feature_variation_dict.keys()):
            # current_row = np.append(current_row, key_level_1[key_1])
            current_row.append(key_level_1[key_1])

            current_row.append(0)
            
            for i in range(len(key_level_2)):
                current_row.append([0,0,0])
            
            rows.append(current_row)
            continue
        
        current_row.append(key_level_1[key_1])
        current_row.append(interaction_wise_count_dict[key_level_1[key_1]])
        current_interaction_feature_wise_variation = interaction_wise_feature_variation_dict[key_level_1[key_1]]
    
        for key_2 in key_level_2:
            current_feature_val = current_interaction_feature_wise_variation[key_2]
            
            current_row.append(current_feature_val)

    
        rows.append(current_row)
    
    
            
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)

    return


def save_csv(fields, rows, filename):

    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(rows)




    return


######Visualization Functions#######


def plot_scatter(data_filename, x_label, y_label, image_filename, no_of_iterations):


    dataset = pd.read_csv(data_filename)
    sns.set_theme(style="darkgrid")
    data_val = dataset['Training Sample']
    cost_val = dataset['Cost']

    iteration_val = dataset['Iteration Number']

    
    indices = []
    line_position = []
    for i in range(no_of_iterations):
        index = np.where(iteration_val == (i+1))[0][0]
        indices.append(index)
        line_position.append(cost_val[index])
    

    
    
    # sns.relplot(x=x_label, y=y_label, data=dataset, hue='Cost', col='Iteration Number', kind="scatter")
    g = sns.FacetGrid(col="Iteration Number" ,data=dataset)
    g.map_dataframe(sns.scatterplot, x = x_label, y = y_label)
    count = 0
    for ax, pos in zip(g.axes.flat, line_position):
        ax.axhline(y=pos, color='r', linestyle=':')
        ax.scatter(data_val[indices[count]],cost_val[indices[count]],facecolors='green',alpha=.5, s=2)
        count += 1
    
    g.savefig(image_filename)


    return



def plot_scatter_power_spectrum(data_dictionary, x_label, y_label, image_filename):

    no_of_regions = data_dictionary["No of regions"]
    sample_size = data_dictionary["Sample Size"]
    cost_vector = data_dictionary["Cost"]
    updated_cost_vector = cost_vector/np.min(cost_vector)
    data_dictionary["Cost"] = updated_cost_vector
    dataset_frame = pd.DataFrame.from_dict(data_dictionary)
    
    max_cost = np.max(cost_vector)
    sns.displot(x = x_label, data = dataset_frame, binwidth = 15)
    plt.text(x=max_cost-20, y=sample_size/4, s = 'No of regions: '+ str(no_of_regions))
    plt.text(x=max_cost-20, y=sample_size/4-3000, s = 'Sample Size: '+ str(sample_size))
    
    # sns.displot(data=dataset_frame, x=x_label, kind="kde", bw_adjust = 2, cut=0)
    plt.savefig(image_filename)
    # plt.show()


    return


def plot_cost_data(cost_data_vec, iteration_no, filename):


    cost_data_dict = {}
    cost_data_dict["cost"] = cost_data_vec
    cost_data_dict["iterations"] = np.arange(1,len(cost_data_vec)+1)

    max_cost = np.max(cost_data_vec)
    min_cost = np.min(cost_data_vec)
    max_iteration = len(cost_data_vec)+1
    cost_data_frame = pd.DataFrame.from_dict(cost_data_dict)


    sns.relplot(x="iterations", y="cost", kind="line", data=cost_data_frame)
    plt.text(x=max_iteration-20, y= 3*(max_cost+min_cost)/2 , s = 'Iteration No: '+ str(iteration_no))
    plt.savefig(filename)


    return

def plot_effort_pref_results(input_filename, output_filename):

    y_label_1 = "Cost"
    x_label_1 = "Starting Region"

    y_label_2 = r'$\eta$'
    x_label_2 = r'$\lambda$'


    dataset = pd.read_csv(input_filename)
    sns.set_theme(style="whitegrid")


    regions = dataset["region"]
    cost_before_training = dataset["cost_1"]
    relative_cost = cost_before_training/np.max(cost_before_training)
    
    eta_value = dataset["cost_2"]
    eta_value = eta_value/np.max(eta_value)

    plotting_data_1 = {}
    plotting_data_1["region"] = regions
    plotting_data_1["cost"] = relative_cost
    
    plotting_data_2 = {}
    eta_value -= np.min(eta_value)
    val = eta_value+relative_cost
    plotting_data_2["eta"] = eta_value+relative_cost
    plotting_data_2["region"] = regions
    font = font_manager.FontProperties(weight='bold', style='normal', size=35)
    plt.figure(1)
    ax1 = plt.subplot(221)
    sns.barplot(data=plotting_data_1 ,x="region", y="cost", palette="bright", label="starting regions")
  
    ax1.set_title("Before Effort-based Penalty", fontsize=20, weight='bold')
    ax1.set_xlabel("Starting Region", fontsize=20, weight='bold')
    ax1.set_ylabel("C($\\xi$)", fontsize=20, weight='bold')
    ax1.set_ylim([0,2.5])
    ax1.annotate(xy=(0,1.1), s="$\\xi_1^e$", fontsize=20, weight='bold')
    ax1.annotate(xy=(1,1.1), s="$\\xi_2^e$",fontsize=20, weight='bold')
    ax1.annotate(xy=(2,1.1), s="$\\xi_3^e$",fontsize=20, weight='bold')
    ax1.annotate(xy=(3,1.1), s="$\\xi^*$",fontsize=20, weight='bold')
   
    
    ax1.tick_params(axis='x', labelsize='20')
    ax1.tick_params(axis='y', labelsize='20')
    plt.subplot(222)
    ax2 = sns.barplot(data=plotting_data_2 ,x="region", y="eta", palette="bright", label="starting regions")
    
    ax2.set_title("After Effort-based Penalty", fontsize=20, weight='bold')
    ax2.set_xlabel("Starting Region", fontsize=20, weight='bold')
    ax2.set_ylabel("C($\\xi$) + $\eta$ ($\\xi$)", fontsize=20, weight='bold')
    ax2.set_ylim([0,2.5])
    ax2.annotate(xy=(0,val[0]+0.1), s="$\\xi_1^e$", fontsize=20, weight='bold')
    ax2.annotate(xy=(1,val[1]+0.1), s="$\\xi_2^e$", fontsize=20, weight='bold')
    ax2.annotate(xy=(2,val[2]+0.1), s="$\\xi_3^e$", fontsize=20, weight='bold')
    ax2.annotate(xy=(3,val[3]+0.1), s="$\\xi^*$", fontsize=20, weight='bold')
    
    ax2.tick_params(axis='x', labelsize='20')
    ax2.tick_params(axis='y', labelsize='20')

    plt.show()
    
    return


def plot_min_term_results(input_filename, output_filename):

    dataset = pd.read_csv(input_filename)
    sns.set_theme(style="whitegrid")

    plotting_data = {}

    # plotting_data["iteration"] = dataset["iteration"]
    # plotting_data["violation"] = [dataset["with_violation"], dataset["without_violation"]]
    dataset_frame = pd.DataFrame.from_dict(dataset)

    g = sns.catplot(data=dataset_frame ,x="iteration", y="violations", hue="cost function", palette="bright",kind="bar", legend=False)
    
    g.set_axis_labels("Number of Updates", "Number of Violations", weight='bold', fontsize=50)
    # g.set_label(loc='upper right')

    font = font_manager.FontProperties(weight='bold', style='normal', size=35)
    plt.tick_params(axis='x', labelsize='50')
    plt.tick_params(axis='y', labelsize='50', length=4)
    plt.legend(loc="upper right", prop=font)


    plt.show()    
    return


def scatter_plot_avg(input_file):

    dataset = pd.read_csv(input_file)
    
    data_val = dataset['Training Sample']
    cost_val = dataset['Cost']

    iteration_val = dataset['Iteration Number']
    indices = []
    line_position = []
    no_of_iterations = 2
    for i in range(no_of_iterations):
        index = np.where(iteration_val == (i+1))[0][0]
        indices.append(index)
        line_position.append(cost_val[index])
    

    
    
    sns.set(font_scale = 2)
    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(col="Iteration Number" ,data=dataset, legend_out=False)
    
    g.map_dataframe(sns.scatterplot, x="Training Sample", y="Cost", alpha=1, s=500, color='tab:blue')
    font = font_manager.FontProperties(weight='bold', style='normal', size=40)
    count = 0
    for ax, pos in zip(g.axes.flat, line_position):
        ax.axhline(y=pos, color='green', linestyle='--',linewidth=6.5)
        
        ax.scatter(data_val[indices[count]],cost_val[indices[count]],facecolors='#008000',alpha=1, s=1200, label="$\\xi^*$", marker='*')
        ax.scatter(data_val[1:4],cost_val[indices[count]+1:indices[count]+4],facecolors='red',alpha=0.9, s=600, label="Min Cost Samples")
        ax.scatter(data_val[1:4],cost_val[indices[count]+1:indices[count]+4],edgecolors='red',facecolors='none', s=700, linestyle="--", linewidth=1)
        ax.tick_params(axis='x', labelsize='40')
        ax.tick_params(axis='y', labelsize='40')
        ax.legend(loc='upper left', prop=font)
        ax.set_xlabel("Training Sample", fontsize = 50, weight='bold')
        ax.set_ylabel("Cost", fontsize = 50, weight='bold')
        ax.annotate(s='Min Cost Line', xy = (5,pos-15), weight='bold', fontsize=35)
        if(count == 0):
            text = "Without Average Cost"
            text = "Update Number 1"
            ax.set_title(text, fontsize=40, weight='bold')
          
        else:
          
            text = "With Average Cost"
            text = "Update Number 2"

            ax.set_title(text, fontsize=40, weight='bold')
            # ax.text(x=0, y=75, s = "$\\xi^*$")
        count += 1
    
    
    plt.show()
    
    
    return