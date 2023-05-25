####This File consists of all functions related to measuring strength of feature interactions for demonstrated trajectories######
#####This should only be tested for real dataset#####################
####Maintainer: Omey Mohan Manyar, Email ID: manyar@usc.edu#############################


import numpy as np
from queue import PriorityQueue
import copy

import utilities as utils
from datastructures import *
from scipy.stats import skew
import itertools
import argparse
import json


###Class to evaluate feature interactions
class feature_interaction:
    def __init__(self, initialization_dict):
        
        self.feature_id_vec = []               #This consists of an array of the feature ids 

        self.no_of_features = 2            #At least two features should be present for detecting interactions

        self.feature_count = {}             #This dictionary will hold the feature count for a certain feature

        self.no_of_demonstrated_trajs = 0       #The total number of demonstrations that have been performed

        self.suspected_l2_interacting_features = []    #An array of size(nx2) depicting feature interactions upto 2 features

        self.suspected_l3_interacting_features = []    #An array of size(nx3) depicting feature interactions upto 3 features

        self.variance_threshold = 0.5                   #This is the threshold used to ensure that data is randomly distributed        

        self.traj_feature_dict = {}                     #This dictionary consists of feature vectors for every state in a trajectory. Key value is trajectory number

        self.feature_level_mapping = {}                #This dictionary consists of feature values High, Medium and Low and maps them to a value

        
        self.trajectory_wise_feature_skewness = {}      #This dictionary consists of trajectory number as the keys and values as array of skewness value of every feature

        
        self.skewness_threshold = 2                     #This is the allowable skewness to determine if the other feature data is constant or not
        self.evaluate_skewness = True                  #This flag will ensure the skewness computation is performed for the trajectories
        
        self.initialize_feature_interactions(initialization_dict)

        return


    ##Method to initialize certain parameters for feature interaction class
    ##The input is an initialization vector
    def initialize_feature_interactions(self, initialization_dict):

        self.no_of_features = copy.deepcopy(initialization_dict['no_of_feature'])

        self.feature_id_vec = copy.deepcopy(initialization_dict['feature_ids'])

        self.no_of_demonstrated_trajs = copy.deepcopy(initialization_dict['no_of_traj'])

        self.traj_feature_dict = copy.deepcopy(initialization_dict['feature_collection'])

        self.suspected_l2_interacting_features = copy.deepcopy(initialization_dict['suspected_l2_interactions'])

        # self.suspected_l3_interacting_features = copy.deepcopy(initialization_dict['suspected_l3_interactions'])

        self.feature_level_mapping['H'] = 2   
        self.feature_level_mapping['M'] = 0
        self.feature_level_mapping['L'] = -2

        
        self.feature_interaction_keys = [''.join(i) for i in itertools.product(self.feature_level_mapping.keys(), repeat = 2)]

        self.map_key_to_index = {}          #Maps the interaction type to an index

        for key in range(len(self.feature_interaction_keys)):
            self.map_key_to_index[self.feature_interaction_keys[key]] = key

        
        ###Vector for differentiating Low, Medium, High Values for respective features###
        self.feature_thresholds = copy.deepcopy(initialization_dict['feature_value_thresholds'])

        ####Using the thresholds generate interaction matrix#####
        self.generate_feature_scale_matrix()

        return


    
    def generate_feature_scale_matrix(self):

        #Iterating through all trajectories
        self.traj_wise_feature_variation = {}             #Array of dictionaries of features for each trajectories
        self.traj_wise_feature_variation_alpha = {}

        self.overall_feature_variation = {}
        self.overall_feature_variation_alpha = {}
        

        for traj in range(self.no_of_demonstrated_trajs):
            
            current_traj_feature_vals = self.traj_feature_dict[str(traj+1)]

            [no_of_features, no_of_regions] = np.shape(current_traj_feature_vals)
            current_traj_feature_variation = {}
            current_traj_feature_skewness = {}
            current_traj_feature_variation_alpha = {}

            for feature in range(no_of_features):
                region_wise_feature_vector = current_traj_feature_vals[feature,:]         #Gets the vector containing feature value for every region
                
                current_feature_thresholds = self.feature_thresholds[self.feature_id_vec[feature]]              #Vector of size 3 with low medium and high value
                current_feature_scale_vec = np.zeros(no_of_regions, dtype=int)
                current_feature_scale_vec_alpha = np.repeat('H', no_of_regions)
                
                ###If the feature value is low
                low_value_indices = np.where(region_wise_feature_vector<=current_feature_thresholds[0])[0]
                current_feature_scale_vec[low_value_indices] = self.feature_level_mapping['L']
                current_feature_scale_vec_alpha[low_value_indices] = 'L'

                ###If the feature value is medium
                medium_value_indices = np.where(np.logical_and(region_wise_feature_vector>current_feature_thresholds[0],region_wise_feature_vector<=current_feature_thresholds[1]))[0]
                current_feature_scale_vec[medium_value_indices] = self.feature_level_mapping['M']
                current_feature_scale_vec_alpha[medium_value_indices] = 'M'

                ###If the feature value is high
                high_value_indices = np.where(region_wise_feature_vector>current_feature_thresholds[1])[0]
                current_feature_scale_vec[high_value_indices] = self.feature_level_mapping['H']
                current_feature_scale_vec_alpha[high_value_indices] = 'H'

                ##Computes the skewness of the feature
                current_feature_skewness = skew(current_feature_scale_vec)
                
                
                current_traj_feature_variation[self.feature_id_vec[feature]] = current_feature_scale_vec
                current_traj_feature_skewness[self.feature_id_vec[feature]] = current_feature_skewness
                current_traj_feature_variation_alpha[self.feature_id_vec[feature]] = current_feature_scale_vec_alpha

                if(self.feature_id_vec[feature] not in self.overall_feature_variation.keys()):
                    self.overall_feature_variation[self.feature_id_vec[feature]] = current_feature_scale_vec
                    self.overall_feature_variation_alpha[self.feature_id_vec[feature]] = current_feature_scale_vec_alpha
                else:
                    old_feature_variation_vec = copy.deepcopy(self.overall_feature_variation[self.feature_id_vec[feature]])
                    old_feature_variation_alpha_vec = copy.deepcopy(self.overall_feature_variation_alpha[self.feature_id_vec[feature]])
                    old_feature_variation_vec = np.append(old_feature_variation_vec, current_feature_scale_vec)
                    old_feature_variation_alpha_vec = np.append(old_feature_variation_alpha_vec,current_feature_scale_vec_alpha)

                    self.overall_feature_variation[self.feature_id_vec[feature]] = old_feature_variation_vec
                    self.overall_feature_variation_alpha[self.feature_id_vec[feature]] = old_feature_variation_alpha_vec


            self.traj_wise_feature_variation[str(traj+1)] = current_traj_feature_variation
            self.trajectory_wise_feature_skewness[str(traj+1)] = current_traj_feature_skewness
            self.traj_wise_feature_variation_alpha[str(traj+1)] = current_traj_feature_variation_alpha
        
        
        self.determine_feature_interaction_count()

        
        return


    def determine_feature_interaction_count(self):
        
        self.l2_no_of_captured_interactions = {}                ##Dictionary consisting of a ratio: no_of_interactions capture/total no of possible interactions
        self.l3_no_of_captured_interactions = {}

        self.l2_feature_interaction_type_priority_dict = {}     ##Dictionary consisting of key as the declared interaction and value is a priority queue of type of interaction


        self.feature_count_dict = {}

        no_of_possible_l2_interations = len(self.feature_interaction_keys)
        self.l2_featurewise_interaction_type_count = {}        #This is a dictionary of feature_interactions with dictionary showing the count for all types of interactions

        self.interaction_type_feature_wise_variation_count = {}
    
        for current_interaction in self.suspected_l2_interacting_features:
            
            current_interaction_key_value = current_interaction[0]+','+current_interaction[1]
            self.l2_featurewise_interaction_type_count[current_interaction_key_value] = dict.fromkeys(self.feature_interaction_keys, 0)
            self.interaction_type_feature_wise_variation_count[current_interaction_key_value] = {} #[L,M,H]
            

            current_interaction_instances = 0
            current_feature_interaction_type_vec = np.zeros(no_of_possible_l2_interations)
            
            current_feature_type_pq = PriorityQueue()

            
            other_feature_skewness_vec = []

            
            feature_1_variation_alpha = self.overall_feature_variation_alpha[current_interaction[0]]
            feature_2_variation_alpha = self.overall_feature_variation_alpha[current_interaction[1]]
            
            current_interaction_instances += len(feature_1_variation_alpha)
            
            ##Getting the skewness of other features
            if(self.evaluate_skewness):
                for feature in self.feature_id_vec:
                    if(feature in current_interaction):
                        continue
                    
                    current_skewness = self.overall_feature_variation[feature] 
                    other_feature_skewness_vec.append(current_skewness)
                
                    
                other_feature_skewness_vec = np.abs(other_feature_skewness_vec)
                max_skewness = np.max(other_feature_skewness_vec)

                if(max_skewness> self.skewness_threshold):
                    continue

                 
            interaction_wise_feature_dict = {}
            for interaction_id in range(len(feature_1_variation_alpha)):
                interaction_type_key = feature_1_variation_alpha[interaction_id] + feature_2_variation_alpha[interaction_id]
                
                current_count = copy.deepcopy(self.l2_featurewise_interaction_type_count[current_interaction_key_value][interaction_type_key])
                current_count += 1
                self.l2_featurewise_interaction_type_count[current_interaction_key_value][interaction_type_key] = copy.deepcopy(current_count)
                current_feature_interaction_type_vec[self.map_key_to_index[interaction_type_key]] = copy.deepcopy(current_count)
                
                if(interaction_type_key not in interaction_wise_feature_dict.keys()):
                    interaction_wise_feature_dict[interaction_type_key] = {}
                    feature_wise_interaction_dict = {}
                else:
                    feature_wise_interaction_dict = copy.deepcopy(interaction_wise_feature_dict[interaction_type_key])
                
                ##Counting for other features
                for feature in self.feature_id_vec:
                    if(feature in current_interaction):
                        
                        continue
                    
                    if(feature not in feature_wise_interaction_dict.keys()):
                        feature_wise_interaction_dict[feature] =  [0,0,0]
                    current_feature_vec = copy.deepcopy(self.overall_feature_variation_alpha[feature])
                    
                    current_interaction_feature_count = copy.deepcopy(feature_wise_interaction_dict[feature])
                    
                    
                    if(current_feature_vec[interaction_id] == 'L'):
                        current_interaction_feature_count[0] += 1
                    elif(current_feature_vec[interaction_id] == 'M'):
                        current_interaction_feature_count[1] += 1
                    elif(current_feature_vec[interaction_id] == 'H'):
                        current_interaction_feature_count[2] += 1
                    
                    feature_wise_interaction_dict[feature] = current_interaction_feature_count

                interaction_wise_feature_dict[interaction_type_key] = feature_wise_interaction_dict
            
            self.interaction_type_feature_wise_variation_count[current_interaction_key_value] = interaction_wise_feature_dict
                    
            
            for current_feature_interaction_type in self.l2_featurewise_interaction_type_count[current_interaction_key_value].keys():
                count = self.l2_featurewise_interaction_type_count[current_interaction_key_value][current_feature_interaction_type]
                current_feature_type_pq.put((count, current_feature_interaction_type))
            
            
            current_feature_instances = len(np.where(current_feature_interaction_type_vec != 0)[0])

            self.l2_feature_interaction_type_priority_dict[current_interaction_key_value] = current_feature_type_pq
            self.l2_no_of_captured_interactions[current_interaction_key_value] = current_feature_instances/no_of_possible_l2_interations
    

        
        print('Interaction wise feature count dictionary: ', self.interaction_type_feature_wise_variation_count)
        

        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Performance-based learning training arguments')
    parser.add_argument('-d', '--data_type', type=str, help='The type of data to train on, synthetic or real')
    parser.add_argument('-t', '--testing_molds', type=list, help='IDs of the molds/tools used as test dataset')
    parser.add_argument('-n', '--dataset_count', type=int, help='Number of files in the dataset')
    parser.add_argument('-s', '--skewness_threshold', type=float, help='Parameter used to decide sknewness threshold. Kappa in the paper')
    parser.add_argument('-p', '--plot_data', type=bool, help='Argument to generate cost plots for performance trainer')
    parser.add_argument('-f', '--feature_count', type=bool, help='Argument to generate cost plots for performance trainer')
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
        dataset_directory = main_directory + '/data/synthetic_data/'
    
    else:
        dataset_directory = main_directory + '/data/real_data/'
    
    data_directory = main_directory + '/data/'

    current_interaction_filename = data_directory + "feature_interaction.json"
    
    interaction_data = utils.read_json(current_interaction_filename)
    top_level_data_keys = ["feature_id", "feature_thresholds", "l2_interactions"]
    corresponding_key_data = [["feature_names"], ["L"],["interacting_feature_1", "interacting_feature_2"]]
    feature_interaction_data = utils.initialize_feature_interaction_data(interaction_data, top_level_data_keys, corresponding_key_data)
    feature_id_vec = feature_interaction_data[0]
    l2_feature_interactions = feature_interaction_data[1]
    feature_thresholds = feature_interaction_data[2]

    
    no_of_molds = 0
    start_index = 1
    skip_data = training_params_dict['testing_molds']
    end_index = training_params_dict['dataset_count']
    demonstrated_trajectories = []
    demonstrated_trajectories_features = {}


    #####Datastructures to store feature values for trajectories######
    feature_information_dict = {}
    traj_id_vec = []
    for i in range (start_index,end_index+1,1):
        if(i in skip_data):
            continue
        no_of_molds += 1
        current_mold_filename = copy.deepcopy(dataset_directory + "Mold_"+str(i)+"_Data.csv")
        print("Filname Passed: ", current_mold_filename)
        
        if(training_params_dict['data_type'] == 'synthetic'):
            current_mold = copy.deepcopy(Graph(current_mold_filename, "synthetic"))
        else:
            current_mold = copy.deepcopy(Graph(current_mold_filename))
        
        print('Done Initializing Mold ', i)
        print('Done adding Mold ', i)

        current_mold_features =  utils.getPathFeatures(current_mold.vert_dict, current_mold.user_preferred_path_num)
        current_mold_features = np.transpose(current_mold_features)
        
        demonstrated_trajectories.append(current_mold.user_preferred_path_num)
        demonstrated_trajectories_features[str(no_of_molds)] = current_mold_features
        if(i == start_index):
            feature_information_dict['traj_id'] = np.repeat(i,len(current_mold.user_preferred_path_num))
            for feature_id in range(len(feature_id_vec)):
                feature_information_dict[feature_id_vec[feature_id]] = np.array(current_mold_features[feature_id])
        
        else:
            traj_id_vec = copy.deepcopy(feature_information_dict['traj_id'])
            traj_id_vec = np.append(traj_id_vec,np.repeat(i,len(current_mold.user_preferred_path_num)))
            feature_information_dict['traj_id'] = traj_id_vec
            for feature_id in range(len(feature_id_vec)):
                current_feature_vec = copy.deepcopy(feature_information_dict[feature_id_vec[feature_id]])
                current_feature_vec = np.append(current_feature_vec, current_mold_features[feature_id])
                feature_information_dict[feature_id_vec[feature_id]] = current_feature_vec

        del(current_mold)

    
    feature_filename = data_directory + "feature_values_num_10.csv"
    utils.save_feature_values(feature_filename, feature_information_dict)
    feature_id_vec = ['internal_undraped', 'internal_draped', 'boundary_undraped', 'boundary_draped', 'region_proximity', 'draped_convexity', 'undraped_convexity', 'curvature', 'neighbor_angle', 'z_height']
    
    input_dictionary = {}       ##This should have information about different type of feature interactions

    input_dictionary['no_of_feature'] = training_params_dict['feature_count']
    input_dictionary['feature_ids'] = feature_id_vec
    input_dictionary['no_of_traj'] = no_of_molds
    input_dictionary['feature_collection'] = demonstrated_trajectories_features

    input_dictionary['suspected_l2_interactions'] = l2_feature_interactions
    
    feature_threshold_dict = {}
    for i in range(len(feature_thresholds)):
        feature_threshold_dict[feature_id_vec[i]] =  feature_thresholds[i]
        
    
    input_dictionary['feature_value_thresholds'] = feature_threshold_dict



    feature_interaction_obj = feature_interaction(input_dictionary)
    feature_interaction_obj.skewness_threshold = training_params_dict['skewness_threshold']
    feature_interaction_dict_alpha ={} 
    feature_interaction_dict_alpha['traj_id'] = traj_id_vec
    feature_interaction_dict_alpha.update(feature_interaction_obj.overall_feature_variation_alpha)
    feature_filename_alpha = data_directory + "feature_values_alpha_10.csv"
    utils.save_feature_values(feature_filename_alpha, feature_interaction_dict_alpha)

    
    for interactions in range(len(feature_interaction_obj.suspected_l2_interacting_features)):
        current_key = feature_interaction_obj.suspected_l2_interacting_features[interactions]
        current_key = current_key[0] + ',' + current_key[1]
        
        level2_keys = set(feature_interaction_obj.suspected_l2_interacting_features[interactions]) ^ set(feature_interaction_obj.feature_id_vec)
        
        feature_interaction_variation_filename = data_directory + current_key + ".csv"
        utils.save_feature_interaction(feature_interaction_variation_filename, feature_interaction_obj.l2_featurewise_interaction_type_count[current_key], feature_interaction_obj.interaction_type_feature_wise_variation_count[current_key], current_key, feature_interaction_obj.feature_interaction_keys, level2_keys)