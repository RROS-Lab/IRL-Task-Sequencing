import numpy as np
from scipy.optimize import minimize
from datastructures import *
import utilities as utils
import argparse
import json




class preference_learner:
    def __init__(self):


        #This dictionary consists of feature vectors for every training data present in the dataset
        #The key value 1,2,3,4,......,N
        #The mapped value is a feature vector consisting of x and y coordinates of starting and ending state
        self.topological_feature_dict = {}


        self.learnt_weights = []

        self.feature_vec_dim = 4        ##By default this is 2 as f(x_s, y_s, x_e,y_e) = n
        self.no_of_molds = 0

        self.starting_preference_weights = np.random.rand(self.feature_vec_dim)
        self.ending_preference_weights = np.random.rand(self.feature_vec_dim)

        ######The dataset based on every tool that is present
        self.start_preference_training_data = {}
        self.end_preference_training_data = {}
        


        self.mold_user_preferred_path = {}
        self.mold_start_region_centroid = {}


        self.keys_of_preference_training_data = ["sampled_paths", "centroid"]

        ###This is a dictionary that holds the sampled weights
        self.start_preference_sample_weights = {}

        self.max_iterations = 3
        self.no_of_current_iterations = 0

        self.mold_data_dict = {}

        self.data_skip = []

        return



    ###This method adds a mold that is the object of the class Graph###
    def add_data(self, mold):

        self.no_of_molds += 1
        self.no_of_molds = int(self.no_of_molds)
        
        self.mold_user_preferred_path[str(self.no_of_molds)] = mold.user_preferred_path_num
        
        start_region_id = utils.num2alpha(mold.user_preferred_path_num[0])
        self.mold_start_region_centroid[str(self.no_of_molds)] = mold.region_information_dict[start_region_id].region_centroid

        self.mold_data_dict[str(self.no_of_molds)] = copy.deepcopy(mold)

        self.generate_start_preference_training_data(mold)

        return

    
    def generate_start_preference_training_data(self, current_mold):

      
        current_user_preferred_path = current_mold.user_preferred_path_num
        cost_of_user_preferred_path = utils.getPathCost(current_mold.vert_dict, current_user_preferred_path)
        user_preferred_start_region = current_user_preferred_path[0]
        user_preferred_start_region_id = utils.num2alpha(user_preferred_start_region)
        user_preferred_start_region_centroid = current_mold.region_information_dict[str(user_preferred_start_region_id)].region_centroid
        
        
        sampled_data = utils.k_similar_paths(current_mold)
        print("Cost of User preffered path: ", current_user_preferred_path, " Cost: ", cost_of_user_preferred_path)
        print("\n")
        
        start_index_array = sampled_data[:,0]
        
   
        unique_start_regions = np.where(current_user_preferred_path[0] != start_index_array)[0]
        unique_start_regions = np.unique(start_index_array[unique_start_regions])
        

        current_mold_training_data = {}
        
        start_region_sample_weights = {}

        for i in range(len(unique_start_regions)):
            current_region_information_dict = {}
            
            current_region_id = unique_start_regions[i]
            current_region_indices = np.where(current_region_id == start_index_array)[0]
            current_region_start_sampled_paths = sampled_data[current_region_indices]
            current_region_information_dict[self.keys_of_preference_training_data[0]] = current_region_start_sampled_paths
            alpha_id = utils.num2alpha(current_region_id)
            current_region_centroids = current_mold.region_information_dict[str(alpha_id)].region_centroid
            current_region_information_dict[self.keys_of_preference_training_data[1]] = current_region_centroids
            current_mold_training_data[str(current_region_id)] = copy.deepcopy(current_region_information_dict)
            
            euclidean_distance = np.linalg.norm(user_preferred_start_region_centroid - current_region_centroids)
            start_region_sample_weights[str(current_region_id)] = euclidean_distance
        
        ###Now add rest of the regions
        no_of_regions = current_mold.no_of_regions
        for i in range(1, no_of_regions+1,1):
            if(str(i) in str(unique_start_regions) or (i == user_preferred_start_region)):
                continue
            
            current_region_information_dict = {}
            current_region_id = i
            current_region_start_sampled_paths = []
            current_region_information_dict[self.keys_of_preference_training_data[0]] = current_region_start_sampled_paths
            alpha_id = utils.num2alpha(current_region_id)
            current_region_centroids = current_mold.region_information_dict[str(alpha_id)].region_centroid
            current_region_information_dict[self.keys_of_preference_training_data[1]] = current_region_centroids
            current_mold_training_data[str(current_region_id)] = copy.deepcopy(current_region_information_dict)
            
            euclidean_distance = np.linalg.norm(user_preferred_start_region_centroid - current_region_centroids)
            start_region_sample_weights[str(current_region_id)] = euclidean_distance



        self.start_preference_sample_weights[str(self.no_of_molds)] = start_region_sample_weights
        self.start_preference_training_data[str(self.no_of_molds)] = current_mold_training_data
        


        return

    ##Cost function to learn preference index
    def start_preference_max_margin_cost(self, weight_vector):

        cost_value = 0.0
        lambda_parameter = 2
        loss_parameter = 1

        total_data_count = 0

        for i in range(1, self.no_of_molds+1,1):
            current_mold_user_preferred_path = self.mold_user_preferred_path[str(i)]

            current_mold_user_preferred_path_start_region = current_mold_user_preferred_path[0]

            current_mold_start_preference_data = self.start_preference_training_data[str(i)]



            if(bool(current_mold_start_preference_data)):
                    
                total_data_count += 1
                user_preferred_start_region_centroids = self.mold_start_region_centroid[str(i)]
                value_of_user_preferred_start = weight_vector[0] * user_preferred_start_region_centroids[0] + weight_vector[1] * (1-user_preferred_start_region_centroids[0]) + \
                weight_vector[2] * user_preferred_start_region_centroids[1] + weight_vector[3] * (1-user_preferred_start_region_centroids[1])
                current_mold_cost = 0
                current_mold_sample_weights = self.start_preference_sample_weights[str(i)]
                
                for region_keys in current_mold_start_preference_data:
                    
                    current_region_dict = current_mold_start_preference_data[region_keys]
                    # print(current_region_dict)
                    sampled_paths = current_region_dict[self.keys_of_preference_training_data[0]]
                    sampled_region_centroid = current_region_dict[self.keys_of_preference_training_data[1]]

                    #####Over here add a function to ask the user if this path is preferred or not
                    ##if(true) continue
                    current_region_cost_value = 0
                    current_region_cost_value = weight_vector[0] * sampled_region_centroid[0] + weight_vector[1] * (1-sampled_region_centroid[0])+ \
                    weight_vector[2] * sampled_region_centroid[1] + weight_vector[3] * (1-sampled_region_centroid[1])

                    current_start_weights = 10*current_mold_sample_weights[region_keys]
                    current_mold_cost += current_start_weights*(value_of_user_preferred_start- current_region_cost_value)
                
            cost_value += current_mold_cost 

        cost_value = cost_value/total_data_count 
        total_cost = loss_parameter * cost_value + lambda_parameter*np.dot(np.transpose(weight_vector), weight_vector)
        self.no_of_current_iterations += 1
        # print("Computed Cost Value: ", total_cost)
        return total_cost

    
    def train_starting_preference(self, weights_init = None, bounds = None):

        ##Generate a dataset of training for starting preference

        converged_flag = False

        ###Initializing the weights
        if(weights_init == None):
            weights_init = 10*np.random.rand(self.feature_vec_dim)
        
        ####Setting the bounds for the optimizer
        if(bounds == None):
            bounds = [[0.1,None]] * self.feature_vec_dim

    

        ####The no of loop iterations completed###
        loop_iterations = 0

        # current_bounds = MyBounds()
        weights = copy.deepcopy(weights_init)
        while(not converged_flag and (loop_iterations < self.max_iterations)):
            # weights_init = np.random.rand(self.mold_graph[str(1)].no_of_features)
            weights_init = weights * np.random.rand(1)
            # result = minimize(self.tiling_cost_function, weights_init, bounds=bounds)
            result = minimize(self.start_preference_max_margin_cost, weights_init, bounds=bounds)
            
            # minimizer_kwargs = {"method":"L-BFGS-B"}
            # result = basinhopping(self.cost_function, weights_init, niter=300, stepsize=2, accept_test=current_bounds, minimizer_kwargs=minimizer_kwargs)
            # result = brute(self.cost_function, ranges=bounds, finish=optimize.fmin)
            weights = copy.deepcopy(result.x)

            print('The training converged for loop iteration: ', self.no_of_current_iterations)
            self.no_of_current_iterations = 0
            print('Value of Weights: ', weights)
            print('Cause for Optimization Termination:', result.message)
            print("Success Flag: ", result.success)


            #####Add the code to update the data#####
            for i in range(1,self.no_of_molds+1,1):
                current_mold_start_preference_data = self.start_preference_training_data[str(i)]
                user_preferred_start_region_centroids = self.mold_start_region_centroid[str(i)]
                value_of_user_preferred_start = weights[0] * user_preferred_start_region_centroids[0] + weights[1] * (1-user_preferred_start_region_centroids[0]) + \
                weights[2] * user_preferred_start_region_centroids[1] + weights[3] * (1-user_preferred_start_region_centroids[1])
                current_mold_cost = 0
                current_mold_sample_weights = self.start_preference_sample_weights[str(i)]
                converged_flag = True
                for region_keys in current_mold_start_preference_data:
                    current_region_dict = current_mold_start_preference_data[region_keys]
                    # print(current_region_dict)
                    sampled_paths = current_region_dict[self.keys_of_preference_training_data[0]]
                    sampled_region_centroid = current_region_dict[self.keys_of_preference_training_data[1]]

                    current_region_cost_value = weights[0] * sampled_region_centroid[0] + weights[1] * (1-sampled_region_centroid[0])+ \
                    weights[2] * sampled_region_centroid[1] + weights[3] * (1-sampled_region_centroid[1])
                    print(" ", current_region_cost_value, ", ", value_of_user_preferred_start)
                    if(current_region_cost_value <= value_of_user_preferred_start):
                        current_sample_weight = copy.deepcopy(current_mold_sample_weights[region_keys])
                        current_sample_weight = 1000*current_sample_weight
                        current_mold_sample_weights[region_keys] = current_sample_weight
                        converged_flag = False
                        print("Yes")

                        


            loop_iterations += 1

        self.starting_preference_weights = weights
        self.test_convergence()
        ##Generate a dataset of training for ending preference


        return

    
    def test_convergence(self):

        weights = copy.deepcopy(self.starting_preference_weights)
        
        for i in range(1,self.no_of_molds+1,1):
            # current_mold_start_preference_data = self.start_preference_training_data[str(i)]
            current_mold_user_preferred_path = self.mold_user_preferred_path[str(i)]
            
            current_mold_user_preferred_path_start_region = current_mold_user_preferred_path[0]
            
            user_preferred_start_region_centroids = self.mold_start_region_centroid[str(i)]
            value_of_user_preferred_start = weights[0] * user_preferred_start_region_centroids[0] + weights[1] * (1-user_preferred_start_region_centroids[0]) + \
            weights[2] * user_preferred_start_region_centroids[1] + weights[3] * (1-user_preferred_start_region_centroids[1])
            
            print("User preferred starting region for Mold ", i, " is ", current_mold_user_preferred_path_start_region, " Cost is: ", value_of_user_preferred_start)
            current_mold_sample_weights = self.start_preference_sample_weights[str(i)]
            converged_flag = True
            current_mold_data =  self.mold_data_dict[str(i)]
            no_of_regions = current_mold_data.no_of_regions

            current_mold_region_information_dict = current_mold_data.region_information_dict

            for region_keys in range(1,no_of_regions+1,1):
                region_id = int(region_keys)
                alpha_id = utils.num2alpha(region_id)
                current_region_dict = current_mold_region_information_dict[alpha_id]
                sampled_region_centroid = current_region_dict.region_centroid

                current_region_cost_value = weights[0] * sampled_region_centroid[0] + weights[1] * (1-sampled_region_centroid[0])+ \
                    weights[2] * sampled_region_centroid[1] + weights[3] * (1-sampled_region_centroid[1])

                print("Current Start region for Mold ", i, " is ", region_keys, " Cost is: ", current_region_cost_value, " Centroid: ", sampled_region_centroid)


            print("\n")


        return



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Effort-based learning training arguments')
    parser.add_argument('-d', '--data_type', type=str, help='The type of data to train on, synthetic or real')
    parser.add_argument('-t', '--testing_molds', type=list, help='IDs of the molds/tools used as test dataset')
    parser.add_argument('-n', '--dataset_count', type=int, help='Number of files in the dataset')
    parser.add_argument('-p', '--plot_data', type=bool, help='Argument to generate cost plots for effort trainer')
    parser.add_argument('-w', '--performance_weights', type=list, help='This consists of the weights that are generated by the performance trainer')
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
        
    if(training_params_dict['data_type'] == 'synthetic'):
        data_directory = main_directory + '/data/synthetic_data/'
    
    else:
        data_directory = main_directory + '/data/real_data/'
    
    plotting_data_directory = os.path.split(main_directory)[0] +'/'

    preference_learner_obj = preference_learner()

    ##################Adding Synthetic Data###########
    no_of_molds = 0
    start_index = 1
    
    skip_data = training_params_dict['testing_molds']
    
    end_index = training_params_dict['dataset_count']
    

    #####The weights learned from performance-based preference learner should be input here
    max_margin_learnt_weights = training_params_dict['performance_weights']
    
    preference_learner_obj.learnt_weights = max_margin_learnt_weights
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
        
        current_mold.reinitialize_weights(max_margin_learnt_weights)
        print('Done Initializing Mold ', i, " with ", current_mold.no_of_regions, " regions")
        preference_learner_obj.add_data(current_mold)
        
        print('Done adding Mold ', i)
        del(current_mold)

    preference_learner_obj.train_starting_preference()
