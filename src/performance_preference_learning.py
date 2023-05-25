####This File consists of the max margin implementation######
####Maintainer: Omey Mohan Manyar, Email ID: manyar@usc.edu#############################

import numpy as np
from scipy import optimize
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import basinhopping
from datastructures import *
import utilities as utils
import json
import argparse

####Class to perform max margin based regression for estimating the weights for reward function####
class max_margin:

    ####Constructor for the max margin class
    def __init__(self):

        #Consists of all the paths required for training
        #The key value is the mold id and the value is the training trajectories
        self.mold_training_dataset = {}

        ##Mold Graph dataset consisting of all the information with respect to a graph
        self.mold_graph = {}
        
        #Consists of the paths that the expert user prefers
        self.mold_user_preferred_path = {}


        ###No of training datasets
        self.no_of_molds = 0

        ###Trajectory weights based on Distance from user preferred
        self.mold_trajectory_weights = {}

        ####Training Parameters####
        self.no_of_sampled_trajectories = {}      ####
        self.no_of_current_iterations = 0
        self.max_iterations = 2

        #####Hyperparameters####
        self.k_shortest_paths = {}
        self.cost_threshold = 100                  ####This is the threshold beyond which the k shortest path will cut off based on cost

        ####Learned Parameters###
        self.learned_weights = []

        self.total_maxmargin_iterations = 0

        ####Visualization Datastructures#####
        self.cost_data_fields = ["Training Sample", "Cost", "Iteration Number"]
        self.cost_data = {}

        self.loss_function_data = []

        self.random_sampling = False

        return

    ###This method adds a mold that is the object of the class Graph###
    def add_data(self, mold):

        self.no_of_molds += 1
        self.no_of_molds = int(self.no_of_molds)

        no_of_regions = copy.deepcopy(mold.no_of_regions)

        ##Sampling a small percentage of trajectories based on the number of regions
        if(no_of_regions == 6):
            self.no_of_sampled_trajectories[str(self.no_of_molds)] = 20
            self.k_shortest_paths[str(self.no_of_molds)] = 3
            
        elif(no_of_regions == 7):
            self.no_of_sampled_trajectories[str(self.no_of_molds)] = 30
            self.k_shortest_paths[str(self.no_of_molds)] = 3
        
        elif(no_of_regions == 8):
            self.no_of_sampled_trajectories[str(self.no_of_molds)] = 30
            self.k_shortest_paths[str(self.no_of_molds)] = 3
        
        elif(no_of_regions == 9):
            self.no_of_sampled_trajectories[str(self.no_of_molds)] = 30
            self.k_shortest_paths[str(self.no_of_molds)] = 3
        
        else:
            self.no_of_sampled_trajectories[str(self.no_of_molds)] = 30
            self.k_shortest_paths[str(self.no_of_molds)] = 3

        ##This is to initialize the feature weight vector##
        ##This assuming that the number of features don't change per data point###
        if(self.no_of_molds == 1):
            self.no_of_features = mold.no_of_features
            self.learned_weights = np.random.rand(mold.no_of_features)

        self.mold_graph[str(self.no_of_molds)] = mold

        self.mold_user_preferred_path[str(self.no_of_molds)] = mold.user_preferred_path_num

        if(self.random_sampling):
            [self.mold_training_dataset[str(self.no_of_molds)], self.mold_trajectory_weights[str(self.no_of_molds)]] = utils.sample_trajectories_random(self.no_of_sampled_trajectories[str(self.no_of_molds)], self.mold_graph[str(self.no_of_molds)])
        else:
            [self.mold_training_dataset[str(self.no_of_molds)], self.mold_trajectory_weights[str(self.no_of_molds)]] = utils.sample_non_similar_trajectories(self.no_of_sampled_trajectories[str(self.no_of_molds)], self.mold_graph[str(self.no_of_molds)])
        
        
        return

    
    def tiling_cost_function(self, weight_vector):
        """_summary_

        Args:
            weight_vector (_type_): Initialized Weight Vector of size equal to number of features

        Returns:
            _type_: Current Total Cost Value
        """

        
        ###########Hyperparameter Values that work the best#################
        # average_cost_factor = 100
        # min_cost_factor = 40
        # lambda_parameter = 100
        ######################################


        average_cost_factor = 8
        min_cost_factor = 1
        shortest_path_cost_factor = 10
        lambda_parameter = 2

        average_cost = 0.0

        max_cost_vector = []
        min_cost_vector = []

        normalization_factor = 0
        cost_function_value = 0
        shortest_path_cost_value_vec = []
        for i in range(1,self.no_of_molds+1,1):

            
            ##Getting the user preferred path
            user_preferred_path = self.mold_user_preferred_path[str(i)]
            ##Updating the weights of the graph
            self.mold_graph[str(i)].reinitialize_weights(weight_vector)
            ##Fetching the cost of the user preferred path
            cost_of_user_preferred_path = utils.getPathCost(self.mold_graph[str(i)].vert_dict, user_preferred_path)
            
            [current_mold_shortest_path, path_previous_nodes] = utils.shortest_path_dijstra(self.mold_graph[str(i)].vert_dict, self.mold_graph[str(i)].source_node)
            [current_mold_shortest_path_solution, current_mold_shortest_path_list, flag] = utils.get_dijstra_result(self.mold_graph[str(i)].source_node, self.mold_graph[str(i)].goal_node, path_previous_nodes)
            current_shortest_path_cost = utils.getPathCost(self.mold_graph[str(i)].vert_dict, current_mold_shortest_path_solution)
            ##Getting the sampled training trajectories
            training_trajectories = self.mold_training_dataset[str(i)]
            trajectory_weights = np.ones(len(training_trajectories))
            
            cost_vector = []
            cost_of_training_samples = []
            for j in range(len(training_trajectories)):
                current_traj = training_trajectories[j]
                cost_of_training_traj = utils.getPathCost(self.mold_graph[str(i)].vert_dict, current_traj)
                
                user_preferred_path_feature_vec = utils.getPathFeatures(self.mold_graph[str(i)].vert_dict ,user_preferred_path)
                current_traj_features_vec = utils.getPathFeatures(self.mold_graph[str(i)].vert_dict ,current_traj)
                sum_of_feature_user_preferred_features = np.sum(user_preferred_path_feature_vec, axis=0)
                sum_of_feature_sample_traj_features = np.sum(current_traj_features_vec, axis=0)
                feature_cosine_similarity = utils.compute_cosine_similarity(sum_of_feature_user_preferred_features, sum_of_feature_sample_traj_features)
                weight_factor = 1 - feature_cosine_similarity
                    
                if(j>self.no_of_sampled_trajectories[str(i)]-1):
                    trajectory_weights[j] = 20000 * weight_factor
                
                online_max_cost = (cost_of_training_traj + 10*weight_factor)
                cost_vector.append((cost_of_training_traj - cost_of_user_preferred_path))
                cost_of_training_samples.append(online_max_cost)
                current_cost = trajectory_weights[j]*(cost_of_training_traj - cost_of_user_preferred_path)
                
                average_cost += current_cost
                normalization_factor += 1
                 
            current_mold_max_cost = np.max(cost_vector)
            current_mold_min_cost = np.min(cost_vector)
            min_cost_vector.append(current_mold_min_cost)
            max_cost_vector.append(current_mold_max_cost)
            
            min_cost_value = np.min(cost_of_training_samples)
            min_cost_index = np.argmin(cost_of_training_samples)
            
            if(np.dtype(min_cost_index) != np.int64):
                min_cost_index = min_cost_index[0]
            current_demo_cost = min_cost_value - cost_of_user_preferred_path
            
            cost_function_value += (trajectory_weights[min_cost_index] * current_demo_cost)
            shortest_path_cost_value_vec.append((current_shortest_path_cost - cost_of_user_preferred_path))
             
        average_cost = average_cost/(normalization_factor)
        cost_function_value = cost_function_value/(self.no_of_molds)
        
        shortest_path_cost_value = np.sum(shortest_path_cost_value_vec)/self.no_of_molds
        
        overall_min_cost = np.min(min_cost_vector)
        
        print('Min Cost Vector: ', min_cost_vector, ' at iteration number: ', self.no_of_current_iterations)
        min_cost_vector = np.asarray(min_cost_vector)
        number_of_violations = len(np.where(min_cost_vector<0)[0])+1
        
        if(overall_min_cost < 0):
            overall_min_cost = number_of_violations*100*overall_min_cost
        
        total_cost = 0.0
        added_trajectories = len(training_trajectories) - (self.no_of_sampled_trajectories[str(i)]-1) + 1
        
        
        total_cost = -average_cost_factor * average_cost + lambda_parameter*np.dot(weight_vector, np.transpose(weight_vector)) - added_trajectories*min_cost_factor*(overall_min_cost) - shortest_path_cost_factor*shortest_path_cost_value
        
        self.loss_function_data.append(total_cost)
        self.no_of_current_iterations +=1

        if(self.no_of_current_iterations %20 == 0):
            print("Cost at Iteration ", self.no_of_current_iterations, " is: ", total_cost)
            print("Average Cost: ", average_cost)
            print('Weights: ', weight_vector)
        
        return total_cost


    def hinge_cost_function(self, weight_vector):

        average_cost = 0.0
        min_cost_factor = 1
        lambda_parameter = 1
        normalization_factor = 0
        
        cost_function_value = 0
        for i in range(1,self.no_of_molds+1,1):

            
            ##Getting the user preferred path
            user_preferred_path = self.mold_user_preferred_path[str(i)]
            ##Updating the weights of the graph
            self.mold_graph[str(i)].reinitialize_weights(weight_vector)
            ##Fetching the cost of the user preferred path
            cost_of_user_preferred_path = utils.getPathCost(self.mold_graph[str(i)].vert_dict, user_preferred_path)
            
            
            ##Getting the sampled training trajectories
            training_trajectories = self.mold_training_dataset[str(i)]
            trajectory_weights = np.ones(len(training_trajectories))
            
            cost_vector = []
            cost_of_training_samples = []
            for j in range(len(training_trajectories)):
                current_traj = training_trajectories[j]
                cost_of_training_traj = utils.getPathCost(self.mold_graph[str(i)].vert_dict, current_traj)
            
                if(j>self.no_of_sampled_trajectories[str(i)]-1):
            
                    user_preferred_path_feature_vec = utils.getPathFeatures(self.mold_graph[str(i)].vert_dict ,user_preferred_path)
                    current_traj_features_vec = utils.getPathFeatures(self.mold_graph[str(i)].vert_dict ,current_traj)
                    sum_of_feature_user_preferred_features = np.sum(user_preferred_path_feature_vec, axis=0)
                    sum_of_feature_sample_traj_features = np.sum(current_traj_features_vec, axis=0)
                    feature_cosine_similarity = utils.compute_cosine_similarity(sum_of_feature_user_preferred_features, sum_of_feature_sample_traj_features)
                    weight_factor = 1 - feature_cosine_similarity
                    trajectory_weights[j] = 20000 * weight_factor
                
                user_preferred_path_feature_vec = utils.getPathFeatures(self.mold_graph[str(i)].vert_dict ,user_preferred_path)
                current_traj_features_vec = utils.getPathFeatures(self.mold_graph[str(i)].vert_dict ,current_traj)
                sum_of_feature_user_preferred_features = np.sum(user_preferred_path_feature_vec, axis=0)
                sum_of_feature_sample_traj_features = np.sum(current_traj_features_vec, axis=0)
                feature_cosine_similarity = utils.compute_cosine_similarity(sum_of_feature_user_preferred_features, sum_of_feature_sample_traj_features)
                weight_factor = 1 - feature_cosine_similarity
                
                current_cost = (cost_of_training_traj + 10*weight_factor)
                cost_vector.append((cost_of_training_traj - cost_of_user_preferred_path))
                cost_of_training_samples.append(current_cost)

                average_cost += current_cost
                normalization_factor += 1

            min_cost_value = np.min(cost_of_training_samples)
            min_cost_index = np.argmin(cost_of_training_samples)
            
            if(np.dtype(min_cost_index) != np.int64):
                min_cost_index = min_cost_index[0]
            current_demo_cost = min_cost_value - cost_of_user_preferred_path
            cost_function_value += (trajectory_weights[min_cost_index] * current_demo_cost)
            
             
        
        cost_function_value = cost_function_value/self.no_of_molds
        
        total_cost = 0.0
        added_trajectories = len(training_trajectories) - (self.no_of_sampled_trajectories[str(i)]-1) + 1
        
        total_cost = lambda_parameter*np.dot(weight_vector, np.transpose(weight_vector)) - added_trajectories*min_cost_factor*(cost_function_value)
        
        self.loss_function_data.append(total_cost)

        self.no_of_current_iterations +=1

        if(self.no_of_current_iterations %20 == 0):
            print("Cost at Iteration ", self.no_of_current_iterations, " is: ", total_cost)
            print('Weights: ', weight_vector)

        return total_cost


    #####Training Methods for evaluating the weights######
    def train_weights(self, weights_init = None, bounds = None):

        ###Initializing the weights
        if(weights_init == None):
            weights_init = np.random.rand(self.mold_graph[str(1)].no_of_features)
        
        ####Setting the bounds for the optimizer
        if(bounds == None):
            bounds = [[0.1,100]] * self.mold_graph[str(1)].no_of_features
        

        print('Initialized Weights: ', weights_init)
        print('Bounds: ', bounds)
        result = minimize(self.cost_function, weights_init, bounds=bounds)
        weights = result.x

        print('Cause for Optimization Termination:', result.message)
        print("Success Flag: ", result.success)
        print('Weights: ', weights)
        print('Total Iterations: ', self.no_of_current_iterations)
        self.learned_weights = weights

        return weights

    ######Function to actively train feature weights for max margin
    def active_training(self, weights_init = None, bounds = None):

        converged_flag = False

        ###Initializing the weights
        if(weights_init == None):
            weights_init = np.random.rand(self.mold_graph[str(1)].no_of_features)
        
        ####Setting the bounds for the optimizer
        if(bounds == None):
            bounds = [[0.1,100]] * self.mold_graph[str(1)].no_of_features

        weights = weights_init

        ####The no of loop iterations completed###
        loop_iterations = 0

        while(not converged_flag and (loop_iterations < self.max_iterations)):

            weights_init = np.random.rand(self.mold_graph[str(1)].no_of_features)*10
            result = minimize(self.cost_function, weights_init, bounds=bounds)
            weights = result.x

            current_convergence = True
            for i in range(1, self.no_of_molds+1, 1):
                self.mold_graph[str(i)].reinitialize_weights(weights)
                source_node = self.mold_graph[str(i)].source_node
                goal_node = self.mold_graph[str(i)].goal_node
                first_k_shortest_paths = utils.k_Shortest_Paths(self.k_shortest_paths[str(i)] ,self.mold_graph[str(i).vert_dict], source_node, goal_node)
                user_preferred_path_cost = utils.getPathCost(self.mold_graph[str(i)].vert_dict, self.mold_user_preferred_path[str(i)])
                current_cost_threshold = user_preferred_path_cost + self.cost_threshold

                if(len(first_k_shortest_paths) == 0):
                    print('System Converged for Mold ', i)
                
                else:

                    for j in range(len(first_k_shortest_paths)):

                        current_k_path_cost = utils.getPathCost(self.mold_graph[str(i)].vert_dict, first_k_shortest_paths[j])

                        if(current_k_path_cost> current_cost_threshold):
                            continue
                        
                        print("Check the Path is Acceptable or not for Mold ", i)
                        # print("Make sure to close the plt window once you've reviewed the path")
                        print("Path: ", first_k_shortest_paths[j])
                        user_response = input("Press 1 if the path is acceptable and 0 if the path is not acceptable: ")
                        
                        if(user_response == str(0)):
                            continue
                        else:
                            current_mold_training_data = copy.deepcopy(self.mold_training_dataset[str(i)])
                            updated_mold_training_data = np.append(current_mold_training_data, [first_k_shortest_paths[j]], axis=0)
                            self.mold_training_dataset[str(i)] = updated_mold_training_data[1000:len(updated_mold_training_data)+1]
                            current_convergence = False
            
            loop_iterations += 1
            converged_flag = current_convergence

        print('System Converged with weights: ', weights)
        print('Total Active Learning Loop Iterations: ', loop_iterations)


    ####Function to perform active learning without any user interaction
    def automated_active_learning(self, weights_init = None, bounds = None):
        converged_flag = False

        ###Initializing the weights
        if(weights_init == None):
            weights_init = 10*np.random.rand(self.mold_graph[str(1)].no_of_features)
        
        ####Setting the bounds for the optimizer
        if(bounds == None):
            bounds = [[0.1,None]] * self.mold_graph[str(1)].no_of_features

    

        ####The no of loop iterations completed###
        loop_iterations = 0

        ###Keys for saving training information
        user_preferred_path_key = "user_preffered_path"
        user_preferred_path_cost_key = "user_preffered_path_cost"
        k_shortest_path_key = 'k_shortest_path'
        k_shortest_path_cost_key = 'K_shortest_path_cost'
        
        training_traj_key = 'training_traj'
        training_traj_cost_key = 'training_traj_cost'
        previous_k_shortest_paths = []

        # current_bounds = MyBounds()
        weights = copy.deepcopy(weights_init)
        while(not converged_flag and (loop_iterations < self.max_iterations)):
            
            # weights_init = np.random.rand(self.mold_graph[str(1)].no_of_features)
            weights_init = weights * np.random.rand(1)
            result = minimize(self.tiling_cost_function, weights_init, bounds=bounds)  
            
            weights = copy.deepcopy(result.x)

            print('The training converged for loop iteration: ', self.no_of_current_iterations)
            self.no_of_current_iterations = 0
            print('Value of Weights: ', weights)
            print('Cause for Optimization Termination:', result.message)
            print("Success Flag: ", result.success)
        

            ##Saving Data
            training_information = {}
            training_information['weights'] = weights

            ####Iteration based cost data#####
            

            current_convergence = True
            for i in range(1, self.no_of_molds+1, 1):
                self.mold_graph[str(i)].reinitialize_weights(weights)
                source_node = self.mold_graph[str(i)].source_node
                goal_node = self.mold_graph[str(i)].goal_node

                first_k_shortest_paths = utils.k_Shortest_Paths(self.k_shortest_paths[str(i)] ,self.mold_graph[str(i)].vert_dict, source_node, goal_node,self.mold_user_preferred_path[str(i)])
                user_preferred_path_cost = utils.getPathCost(self.mold_graph[str(i)].vert_dict, self.mold_user_preferred_path[str(i)])
                
                if(loop_iterations == 0):
                    current_mold_cost_data_vec = []
                    current_mold_cost_data_vec = [[1,user_preferred_path_cost,loop_iterations+1]]
                    current_index = len(current_mold_cost_data_vec)
                else:
                    current_mold_cost_data_vec = copy.deepcopy(self.cost_data[str(i)])
                    current_index = len(current_mold_cost_data_vec)
                    current_mold_cost_data_vec.append([current_index+1,user_preferred_path_cost,loop_iterations+1])
                    current_index = len(current_mold_cost_data_vec)

                if(loop_iterations == 0 and i == 1):
                    previous_k_shortest_paths = copy.deepcopy(first_k_shortest_paths)
                elif(i == 1 and loop_iterations != 0):
                    indice = np.where(self.mold_training_dataset[str(i)] == previous_k_shortest_paths)[0]
                    print('Indices are: ', indice)
                    print('New ones: ', first_k_shortest_paths)
                    print('Old ones: ', previous_k_shortest_paths[indice])

                print('User preferred path for mold ', i, ', Path: ', self.mold_user_preferred_path[str(i)], ' Cost: ', user_preferred_path_cost)
                    
                training_information[user_preferred_path_key] = self.mold_graph[str(i)].user_preferred_path_alpha
                training_information[user_preferred_path_cost_key] = user_preferred_path_cost

                
                
                k_Shortest_paths_alpha = []
                for k in range(len(first_k_shortest_paths)):
                    current_path = utils.convert_num_path_to_alpha_path(first_k_shortest_paths[k])
                    k_Shortest_paths_alpha.append(current_path)
                

                training_information[k_shortest_path_key] = k_Shortest_paths_alpha
                cost_of_k_paths = []


                #####K Shortest Paths Evaluation####
                for j in range(len(first_k_shortest_paths)):

                    current_k_path_cost = utils.getPathCost(self.mold_graph[str(i)].vert_dict, first_k_shortest_paths[j])
                    cost_of_k_paths.append(current_k_path_cost)
                    current_mold_cost_data_vec.append([current_index+j+1, current_k_path_cost, loop_iterations+1])
                
                    print(i, 'th shortest path: ', first_k_shortest_paths[j], ' Cost: ', current_k_path_cost)
                    current_mold_training_data = copy.deepcopy(self.mold_training_dataset[str(i)])
                    print('Current length of dataset: ', len(current_mold_training_data))
                    print(self.mold_training_dataset[str(i)])
                    user_preferred_path_onehot = utils.convert_regionid_to_onehotvector(self.mold_graph[str(i)].user_preferred_path_num)
                    sampled_path_onehot = utils.convert_regionid_to_onehotvector(first_k_shortest_paths[j])
                    similarity_flag = utils.check_if_traj_similar_cosine(self.mold_graph[str(i)].vert_dict, user_preferred_path_onehot, sampled_path_onehot,1e-10)
                    if(str(first_k_shortest_paths[j]) not in str(current_mold_training_data) and (not similarity_flag)):
                        print('Dataset Updated')
                        updated_mold_training_data = np.append(current_mold_training_data, [first_k_shortest_paths[j]], axis=0)
                        self.mold_training_dataset[str(i)] = updated_mold_training_data
                        print('Updated length of dataset: ', len(self.mold_training_dataset[str(i)]))
                        print(self.mold_training_dataset[str(i)])
                    
                
                if(user_preferred_path_cost > cost_of_k_paths[0]):
                    current_convergence = False
                

                ###Evaluating and Saving Training Data###
                current_mold_training_data = copy.deepcopy(self.mold_training_dataset[str(i)])
                training_traj_vec_alpha = []
                training_traj_cost_vec = []
                all_traj_cost_vec = []

                cost_data_start_index = len(current_mold_cost_data_vec)
                for k in range(0,len(current_mold_training_data),1):
                    
                    current_training_traj = current_mold_training_data[k]
                    cost_of_current_training_traj = utils.getPathCost(self.mold_graph[str(i)].vert_dict, current_training_traj)
                    if(k>(self.no_of_sampled_trajectories[str(i)]-1)):
                        current_training_traj_alpha = utils.convert_num_path_to_alpha_path(current_training_traj)
                        training_traj_vec_alpha.append(current_training_traj_alpha)
                        training_traj_cost_vec.append(cost_of_current_training_traj)
                    else:
                        current_mold_cost_data_vec.append([cost_data_start_index+k+1, cost_of_current_training_traj, loop_iterations+1])

                    all_traj_cost_vec.append(cost_of_current_training_traj)
                    
                
                training_information[training_traj_key] = training_traj_vec_alpha
                training_information[training_traj_cost_key] = training_traj_cost_vec
                all_traj_cost_vec = np.sort(all_traj_cost_vec)
                training_information['min_cost_vec'] = all_traj_cost_vec[0:30]
                self.cost_data[str(i)] = current_mold_cost_data_vec
                
                
                training_information['length_of_dataset'] = len(self.mold_training_dataset[str(i)])
                training_information[k_shortest_path_cost_key] = cost_of_k_paths
                
                
                utils.save_training_data(training_information, i, loop_iterations)
            loop_iterations += 1

            for keys in self.cost_data.keys():
                current_mold_filename = "mold_"+ keys + "_cost_data.csv"
                utils.save_csv(self.cost_data_fields, self.cost_data[keys], current_mold_filename)

            
            cost_function_plot_filename = "cost_plot_iteration_"+str(loop_iterations)+".png"
            utils.plot_cost_data(self.loss_function_data, loop_iterations, cost_function_plot_filename)
            self.loss_function_data.clear()
            print('Loop Iterations: ', loop_iterations, '\n')
            self.total_maxmargin_iterations = loop_iterations
            converged_flag = current_convergence

        print('System Converged with weights: ', weights)
        print('Total Active Learning Loop Iterations: ', loop_iterations)



        return weights

    
    def custom_cost_function(self, weight_vector):

        average_cost = 0.0
        
        average_cost_factor = 3
        min_cost_factor = 2
        shortest_path_cost_factor = 8
        lambda_parameter = 1

        

        max_cost_vector = []
        min_cost_vector = []

        normalization_factor = 0
        cost_function_value = 0
        shortest_path_cost_value_vec = []
        for i in range(1,self.no_of_molds+1,1):

            
            ##Getting the user preferred path
            user_preferred_path = self.mold_user_preferred_path[str(i)]
            ##Updating the weights of the graph
            self.mold_graph[str(i)].reinitialize_weights(weight_vector)
            ##Fetching the cost of the user preferred path
            cost_of_user_preferred_path = utils.getPathCost(self.mold_graph[str(i)].vert_dict, user_preferred_path)
            
            ####Getting the shortest paths
            [current_mold_shortest_path, path_previous_nodes] = utils.shortest_path_dijstra(self.mold_graph[str(i)].vert_dict, self.mold_graph[str(i)].source_node)
            [current_mold_shortest_path_solution, current_mold_shortest_path_list, flag] = utils.get_dijstra_result(self.mold_graph[str(i)].source_node, self.mold_graph[str(i)].goal_node, path_previous_nodes)
            current_shortest_path_cost = utils.getPathCost(self.mold_graph[str(i)].vert_dict, current_mold_shortest_path_solution)
            
            ##Getting the sampled training trajectories
            training_trajectories = self.mold_training_dataset[str(i)]
            
            
            weight_array = np.asarray(copy.deepcopy(self.mold_trajectory_weights[str(i)]), dtype=float)
        
            cost_vector = []
            cost_of_training_samples = []
            for j in range(len(training_trajectories)):
                current_traj = training_trajectories[j]
                cost_of_training_traj = utils.getPathCost(self.mold_graph[str(i)].vert_dict, current_traj)
                # print('Cost of User Preferred Path: ', user_preferred_path)
                # print('Cost of current sampled trajectory: ', current_traj, '\n')
                user_preferred_path_feature_vec = utils.getPathFeatures(self.mold_graph[str(i)].vert_dict ,user_preferred_path)
                current_traj_features_vec = utils.getPathFeatures(self.mold_graph[str(i)].vert_dict ,current_traj)
                sum_of_feature_user_preferred_features = np.sum(user_preferred_path_feature_vec, axis=0)
                sum_of_feature_sample_traj_features = np.sum(current_traj_features_vec, axis=0)
                feature_cosine_similarity = utils.compute_cosine_similarity(sum_of_feature_user_preferred_features, sum_of_feature_sample_traj_features)

                cost_vector.append((cost_of_training_traj - cost_of_user_preferred_path))
                current_cost = weight_array[j]*(cost_of_training_traj - cost_of_user_preferred_path)
                cost_of_training_samples.append((cost_of_training_traj - cost_of_user_preferred_path))
                # cost_vector.append((cost_of_training_traj - cost_of_user_preferred_path))
                
                average_cost += current_cost
                normalization_factor += 1
                 
            current_mold_max_cost = np.max(cost_vector)
            current_mold_min_cost = np.min(cost_vector)
            min_cost_vector.append(current_mold_min_cost)
            max_cost_vector.append(current_mold_max_cost)
            
            min_cost_value = np.min(cost_of_training_samples)
            min_cost_index = np.argmin(cost_of_training_samples)
            # print("Min cost index: ", min_cost_index)
            # print("Min cost: ", min_cost_value)
            if(np.dtype(min_cost_index) != np.int64):
                min_cost_index = min_cost_index[0]
            current_demo_cost = current_mold_min_cost - cost_of_user_preferred_path
            # print("Current Demo Cost: ", current_demo_cost)
            cost_function_value += (weight_array[min_cost_index] * current_demo_cost)
            # cost_function_value += (current_demo_cost)
            shortest_path_cost_value_vec.append((current_shortest_path_cost - cost_of_user_preferred_path))
             
        average_cost = average_cost/(normalization_factor)
        cost_function_value = cost_function_value/(self.no_of_molds)
        
        shortest_path_cost_value = np.sum(shortest_path_cost_value_vec)/self.no_of_molds
        
        overall_min_cost = np.min(min_cost_vector)
        
        print('Min Cost Vector: ', min_cost_vector, ' at iteration number: ', self.no_of_current_iterations)
        # print('Overall Min Cost: ', overall_min_cost)
        # print("Shortest Path Vec: ", shortest_path_cost_value_vec)
        min_cost_vector = np.asarray(min_cost_vector)
        number_of_violations = len(np.where(min_cost_vector<0)[0])+1
        
        if(overall_min_cost < 0):
            overall_min_cost = number_of_violations*100*overall_min_cost
        
        total_cost = 0.0
        added_trajectories = len(training_trajectories) - (self.no_of_sampled_trajectories[str(i)]-1) + 1
        
        
        total_cost = -average_cost_factor * average_cost + lambda_parameter*np.dot(weight_vector, np.transpose(weight_vector)) - added_trajectories*min_cost_factor*(overall_min_cost) - shortest_path_cost_factor*shortest_path_cost_value
        
        self.loss_function_data.append(total_cost)
        self.no_of_current_iterations +=1


        ######Update Training Dataset
        self.update_training_data(weight_vector)
        
        print("Cost at Iteration ", self.no_of_current_iterations, " is: ", total_cost)
        print("Average Cost: ", average_cost)
        print('Weights: ', weight_vector)

        return total_cost
    
    
    ####Function to perform active learning without any user interaction
    def performance_pref_active_learning(self, weights_init = None, bounds = None):
        converged_flag = False

        ###Initializing the weights
        if(weights_init == None):
            weights_init = 10*np.random.rand(self.mold_graph[str(1)].no_of_features)
        
        ####Setting the bounds for the optimizer
        if(bounds == None):
            bounds = [[0.1,None]] * self.mold_graph[str(1)].no_of_features

        # current_bounds = MyBounds()
        weights = copy.deepcopy(weights_init)
    
            
        # weights_init = np.random.rand(self.mold_graph[str(1)].no_of_features)
        weights_init = weights * np.random.rand(1)
        result = minimize(self.custom_cost_function, weights_init, bounds=bounds, options={'maxiter':200})
        # result = minimize(self.hinge_cost_function, weights_init, bounds=bounds)
        
        # minimizer_kwargs = {"method":"L-BFGS-B"}
        # result = basinhopping(self.cost_function, weights_init, niter=300, stepsize=2, accept_test=current_bounds, minimizer_kwargs=minimizer_kwargs)
        # result = brute(self.cost_function, ranges=bounds, finish=optimize.fmin)
        weights = copy.deepcopy(result.x)

        print('The training converged for loop iteration: ', self.no_of_current_iterations)
        
        self.no_of_current_iterations = 0
        print('Value of Weights: ', weights)
        print('Cause for Optimization Termination:', result.message)
        print("Success Flag: ", result.success)
    

        return weights


    def update_training_data(self, weights):

        for i in range(1, self.no_of_molds+1, 1):
            # self.mold_graph[str(i)].reinitialize_weights(weights)
            source_node = self.mold_graph[str(i)].source_node
            goal_node = self.mold_graph[str(i)].goal_node

            first_k_shortest_paths = utils.k_Shortest_Paths(self.k_shortest_paths[str(i)] ,self.mold_graph[str(i)].vert_dict, source_node, goal_node,self.mold_user_preferred_path[str(i)])
            user_preferred_path_cost = utils.getPathCost(self.mold_graph[str(i)].vert_dict, self.mold_user_preferred_path[str(i)])
            
            
            k_Shortest_paths_alpha = []
            for k in range(len(first_k_shortest_paths)):
                current_path = utils.convert_num_path_to_alpha_path(first_k_shortest_paths[k])
                k_Shortest_paths_alpha.append(current_path)
            

            
            cost_of_k_paths = []


            #####K Shortest Paths Evaluation####
            for j in range(len(first_k_shortest_paths)):

                current_k_path_cost = utils.getPathCost(self.mold_graph[str(i)].vert_dict, first_k_shortest_paths[j])
                cost_of_k_paths.append(current_k_path_cost)
                
                
                current_mold_training_data = copy.deepcopy(self.mold_training_dataset[str(i)])
                print('Current length of dataset: ', len(current_mold_training_data))
                # print(self.mold_training_dataset[str(i)])
                user_preferred_path_onehot = utils.convert_regionid_to_onehotvector(self.mold_graph[str(i)].user_preferred_path_num)
                sampled_path_onehot = utils.convert_regionid_to_onehotvector(first_k_shortest_paths[j])
                similarity_flag = utils.check_if_traj_similar_cosine(self.mold_graph[str(i)].vert_dict, user_preferred_path_onehot, sampled_path_onehot,1e-6)
                user_preferred_path_features = utils.getPathFeatures(self.mold_graph[str(i)].vert_dict, self.mold_graph[str(i)].user_preferred_path_num)
                user_preferred_path_features_vec = np.sum(user_preferred_path_features, axis=0)
                if(str(first_k_shortest_paths[j]) not in str(current_mold_training_data) and (not similarity_flag)):
                    print('Dataset Updated')
                    updated_mold_training_data = np.append(current_mold_training_data, [first_k_shortest_paths[j]], axis=0)
                    current_weight_of_trajs = copy.deepcopy(self.mold_trajectory_weights[str(i)])
                    current_k_path_features = utils.getPathFeatures(self.mold_graph[str(i)].vert_dict, first_k_shortest_paths[j])
                    current_k_path_features_vec = np.sum(current_k_path_features, axis=0)
                    weight_update = utils.compute_cosine_similarity(user_preferred_path_features_vec ,current_k_path_features_vec)
                    # print(weight_update)
                    weight_update = 200000*(1- weight_update)
                    # print(weight_update)
                    current_weight_of_trajs = np.append(current_weight_of_trajs, weight_update)
                    self.mold_training_dataset[str(i)] = updated_mold_training_data
                    self.mold_trajectory_weights[str(i)] = copy.deepcopy(current_weight_of_trajs)

                    # print('Updated length of dataset: ', len(self.mold_training_dataset[str(i)]))
                    print("Update weight vector: ", self.mold_trajectory_weights[str(i)])
                    # print(self.mold_training_dataset[str(i)])


        return
   
   
class MyBounds: 

    def __init__(self, xmax=[0.1,100], xmin=[0.1,100] ):
        

        self.xmax = np.array(xmax)

        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):

        x = kwargs["x_new"]

        tmax = bool(np.all(x <= self.xmax))

        tmin = bool(np.all(x >= self.xmin))

        return tmax and tmin

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performance-based learning training arguments')
    parser.add_argument('-n', '--max_iterations', type=str, help='The number of maximum dataset aggregations that can be performed')
    parser.add_argument('-d', '--data_type', type=str, help='The type of data to train on, synthetic or real')
    parser.add_argument('-t', '--testing_molds', type=list, help='IDs of the molds/tools used as test dataset')
    parser.add_argument('-c', '--dataset_count', type=int, help='Number of files in the dataset')
    parser.add_argument('-p', '--plot_data', type=bool, help='Argument to generate cost plots for performance trainer')
    parser.add_argument('-i', '--training_param_file', required=True, type=str, help='The json filename of the training parameters')


    args = parser.parse_args()
    arg_parse_dict = vars(args)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    print('Current Directory: ', current_directory)
    main_directory = os.path.split(current_directory)[0]
    print('Main Directory: ', main_directory)

    training_filename = main_directory + '/config/' + args.training_param_file
    
    ##Defining Network Architecture Specific Params
    with open(training_filename, 'rb') as file:
        training_params_dict = json.load(file)
        
    max_margin_obj = max_margin()

    if(training_params_dict['data_type'] == 'synthetic'):
        data_directory = main_directory + '/data/synthetic_data/'
    
    else:
        data_directory = main_directory + '/data/real_data/'
    
    plotting_data_directory = os.path.split(main_directory)[0] +'/'
    

    ##################Adding Data###########
    no_of_molds = 0
    start_index = 1
    skip_data = training_params_dict['testing_molds']
    end_index = training_params_dict['dataset_count']
    max_margin_obj.max_iterations = training_params_dict['max_iterations']
    for i in range (start_index,end_index+1,1):
        if(i in skip_data):
            continue
        no_of_molds += 1
        
        current_window_filename = copy.deepcopy(data_directory + "Mold_"+str(i))+"_Data.csv"
        print("Filname Passed: ", current_window_filename)
      
        if(training_params_dict['data_type'] == 'synthetic'):
            current_mold = copy.deepcopy(Graph(current_window_filename, "synthetic"))
        else:
            current_mold = copy.deepcopy(Graph(current_window_filename))
        
        print('Done Initializing Mold ', i)
        max_margin_obj.add_data(current_mold)
        print('Done adding Mold ', i)
        del(current_mold)

    #############Training for the added data########################
    # max_margin_obj.train_weights()
    start = timeit.default_timer()
    max_margin_obj.automated_active_learning()
    stop = timeit.default_timer()
    print('Total Time for Training : ', stop - start)
    

    if(args.plot_data):
        for i in range (1,no_of_molds+1,1):
            current_mold_filename = copy.deepcopy(plotting_data_directory + "mold_"+str(i)+"_cost_data.csv")
            image_filename = "cost_data_" + str(i) +".png"
            utils.plot_scatter(current_mold_filename, 'Training Sample', 'Cost', image_filename,max_margin_obj.total_maxmargin_iterations)
            