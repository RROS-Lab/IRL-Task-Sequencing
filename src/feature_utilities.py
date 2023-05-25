####This File consists of all functions related to evaluating features for training######
####Maintainer: Omey Mohan Manyar, Email ID: manyar@usc.edu#############################


####Imports######
from os import stat
import numpy as np
import copy
import utilities as utils
################


##Function to evaluate individual features
##Returns a feature vector
def evaluate_features(source, destination, region_features_dict, number_of_features = 10):
    feature_vector = np.zeros(number_of_features)
    no_of_regions = len(source)
    ##Converting Source Vector to approriate one hot encoding
    source_vec = copy.deepcopy(np.array(source))
    source_vec = source_vec.astype(int)
    ##Getting the Draped and Undraped regions in Source State
    source_draped_regions = np.where(source_vec == 1)[0]
    source_draped_regions = source_draped_regions + 1
    source_draped_regions = source_draped_regions.astype(int)

    source_undraped_regions = np.where(source_vec == 0)[0]
    source_undraped_regions = source_undraped_regions + 1
    source_undraped_regions = source_undraped_regions.astype(int)

    ##Converting Destination Vector to approriate one hot encoding
    destination_vec = copy.deepcopy(np.array(destination))
    destination_vec = destination_vec.astype(int)
    ##Getting the Draped and Undraped regions in Destination State
    destination_draped_regions = np.where(destination_vec == 1)[0]
    destination_draped_regions = destination_draped_regions + 1
    destination_draped_regions = destination_draped_regions.astype(int)

    destination_undraped_regions = np.where(destination_vec == 0)[0]
    destination_undraped_regions = destination_undraped_regions + 1
    destination_undraped_regions = destination_undraped_regions.astype(int)
    
    
    action_vec = destination_vec - source_vec

    action = np.where(action_vec == 1)[0]
    action = int(action[0]) + 1
    action_id = utils.num2alpha(action)
    # print("Action ID: ", action_id)
    region_information_dict = copy.deepcopy(region_features_dict)

    ##Finding the maximum z value among all regions:
    [overall_z_max, alpha_region_id_zmax] = get_heighest_zmax(region_information_dict, no_of_regions)
    average_z_height_draped_region = get_average_z_height(region_information_dict, source_draped_regions)
    
    ##This is an object of the class region_features from datastructures.py
    action_region_features = region_information_dict[str(action_id)]

    ###Defining the Features

    ##Feature 1 is the length of the internal edges that are Undraped
    feature_1 = evaluate_feature_internal_free(region_information_dict, source_undraped_regions, action_id)

    ##Feature 2 is the length of the internal edges that are Draped
    feature_2 = evaluate_feature_internal_constrained(region_information_dict, source_draped_regions, action_id)

    ##Feature 3 is the length of the boundary edges that are Undraped
    ##Feature 4 is the length of the boundary edges that are Draped
    [feature_3, feature_4] = evaluate_boundary_opp_status(region_information_dict, action_id, source_undraped_regions, source_draped_regions)

    ##Feature 5 is the distance between currently draped region and the already draped region
    feature_5 = evaluate_region_proximity(region_information_dict, action_id, source_draped_regions)

    ##Feature 6 is the convexity of draped region in the source
    feature_6 = evaluate_convexity(region_information_dict, source)
    

    ##Feature 7 is the convexity of undraped region in the source
    feature_7 = evaluate_convexity(region_information_dict, source, False)

    ##Feature 8 will account for curvature interaction
    feature_8 = curvature_feature(region_information_dict, source_draped_regions,action_id)       

    ##Feature 9 will account for the angle interaction
    feature_9 = neighbor_angle_feature(region_information_dict, action_id, source_draped_regions)


    ##Feature 10 will account for z height interactions
    feature_10 = z_height_feature(region_information_dict, action_id, source_draped_regions) 

    feature_vector = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]

    return feature_vector

###Defining all features

# ##Feature is counting the number of internal edges that are undraped
def evaluate_feature_internal_undraped(region_information_dict, undraped_region_list):

    ##Length of Undraped Internal Edges
    feature_value = 0.0
    
    ##Check if a pair has already been evaluated
    visited_pairs = []

    # print('Undraped Regions: ', undraped_region_list)
    for region_id in undraped_region_list:
        
        alpha_region_id = utils.num2alpha(int(region_id))
        # print('Current Region ID: ', alpha_region_id)
        current_region_information = region_information_dict[str(alpha_region_id)]
        current_edgewise_neighbors = current_region_information.edgewise_region_neighbors

        for i in range(current_region_information.no_of_edges):
            neighbor_id = current_edgewise_neighbors[str(i)]

            for neighbor in neighbor_id:
                current_pair = [str(alpha_region_id), str(neighbor)]
                if(neighbor != 'O' and (str(current_pair) not in str(visited_pairs))):
                    
                    visited_pairs.append([str(alpha_region_id), str(neighbor)])
                    visited_pairs.append([str(neighbor), str(alpha_region_id)])

                    region_dim = current_region_information.region_neighbor_dims[str(neighbor)]
                    feature_value += region_dim
                    
    
    return feature_value

##Feature represents counting the number of internal edges that are undraped
def evaluate_feature_internal_free(region_information_dict, undraped_region_list, region_id):

    ##Length of Undraped Internal Edges
    feature_value = 0.0
    
    current_region_information = region_information_dict[str(region_id)]
    current_edgewise_neighbors = current_region_information.edgewise_region_neighbors
    
    for i in range(current_region_information.no_of_edges):
        neighbor_id = current_edgewise_neighbors[str(i)]

        for neighbor in neighbor_id:
            
            neighbor_num_id = utils.alpha2num(neighbor)
            if(neighbor != 'O' and (neighbor_num_id in undraped_region_list)):
                region_dim = current_region_information.region_neighbor_dims[str(neighbor)]
                feature_value += region_dim
                    
    
    
    perimeter_of_region = current_region_information.region_perimeter
    feature_value = feature_value/perimeter_of_region

    return feature_value


##Feature is counting the number of internal edges that are undraped
def evaluate_feature_internal_draped(region_information_dict, draped_region_list):

    ##Length of Undraped Internal Edges
    feature_value = 0.0
    
    ##Check if a pair has already been evaluated
    visited_pairs = []

    # print('Draped Regions: ', draped_region_list)
    for region_id in draped_region_list:
        
        alpha_region_id = utils.num2alpha(int(region_id))
        # print('Current Region ID: ', alpha_region_id)
        current_region_information = region_information_dict[str(alpha_region_id)]
        current_edgewise_neighbors = current_region_information.edgewise_region_neighbors

        for i in range(current_region_information.no_of_edges):
            neighbor_id = current_edgewise_neighbors[str(i)]

            for neighbor in neighbor_id:
                current_pair = [str(alpha_region_id), str(neighbor)]
                
                if(neighbor != 'O' and (str(current_pair) not in str(visited_pairs))):
                    
                    visited_pairs.append([str(alpha_region_id), str(neighbor)])
                    visited_pairs.append([str(neighbor), str(alpha_region_id)])

                    
                    region_dim = current_region_information.region_neighbor_dims[str(neighbor)]
                    feature_value += region_dim
    

    
    return feature_value

##Feature is counting the number of internal edges that are undraped
def evaluate_feature_internal_constrained(region_information_dict, draped_region_list, region_id):

    ##Length of Undraped Internal Edges
    feature_value = 0.0
 
    current_region_information = region_information_dict[str(region_id)]
    current_edgewise_neighbors = current_region_information.edgewise_region_neighbors
    
    for i in range(current_region_information.no_of_edges):
        neighbor_ids = current_edgewise_neighbors[str(i)]

        for neighbor in neighbor_ids:
            neighbor_num_id = utils.alpha2num(neighbor)
            
            if(neighbor != 'O' and (neighbor_num_id in draped_region_list)):
                region_dim = current_region_information.region_neighbor_dims[str(neighbor)]    
                feature_value += region_dim            
    
    perimeter_of_region = current_region_information.region_perimeter
    feature_value = feature_value/perimeter_of_region
    
    return feature_value


###Function to evaluate the edges opposite a boundary edge being undraped or draped
###This implementation assumes that every region will have exactly 4 edges
def evaluate_boundary_opp_status(region_information_dict, action_id, undraped_region_list, draped_region_list):

    ###Length of the boundary edges for which opposite region is undraped
    feature_val_undraped = 0.0

    ###Length of the boundary edges for which opposite region is draped
    feature_val_draped = 0.0
    
    current_region_information = region_information_dict[str(action_id)]
    current_edgewise_neighbors = current_region_information.edgewise_region_neighbors

    
    for i in range(current_region_information.no_of_edges):
        current_edge_neighbor_ids = current_edgewise_neighbors[str(i)]

        for neighbor in current_edge_neighbor_ids:
            
            if(neighbor == 'O'):
                #Along first edge
                if(i == 0):
                    neighbors_of_edge_3 = current_region_information.edgewise_region_neighbors[str(2)]
                    internal_region_ids = np.where(neighbors_of_edge_3 !='O')[0]
                    
                    for ids in internal_region_ids:
                        num_region_id = utils.alpha2num(str(neighbors_of_edge_3[ids]))
                        region_dim = current_region_information.region_neighbor_dims[str(neighbors_of_edge_3[ids])]
                        if(num_region_id in undraped_region_list):
                            feature_val_undraped += region_dim
                        elif(num_region_id in draped_region_list):
                            feature_val_draped += region_dim

                elif(i == 1):
                    neighbors_of_edge_4 = current_region_information.edgewise_region_neighbors[str(3)]
                    internal_region_ids = np.where(neighbors_of_edge_4 !='O')[0]
                    
                    for ids in internal_region_ids:
                        num_region_id = utils.alpha2num(str(neighbors_of_edge_4[ids]))
                        region_dim = current_region_information.region_neighbor_dims[str(neighbors_of_edge_4[ids])]
                        if(num_region_id in undraped_region_list):
                            feature_val_undraped += region_dim
                        elif(num_region_id in draped_region_list):
                            feature_val_draped += region_dim

                elif(i == 2):
                    neighbors_of_edge_1 = current_region_information.edgewise_region_neighbors[str(0)]
                    internal_region_ids = np.where(neighbors_of_edge_1 !='O')[0]
                    
                    for ids in internal_region_ids:
                        num_region_id = utils.alpha2num(str(neighbors_of_edge_1[ids]))
                        region_dim = current_region_information.region_neighbor_dims[str(neighbors_of_edge_1[ids])]
                        if(num_region_id in undraped_region_list):
                            feature_val_undraped += region_dim
                        elif(num_region_id in draped_region_list):
                            feature_val_draped += region_dim
                
                elif(i == 3):
                    neighbors_of_edge_2 = current_region_information.edgewise_region_neighbors[str(1)]
                    internal_region_ids = np.where(neighbors_of_edge_2 !='O')[0]
                    
                    for ids in internal_region_ids:
                        num_region_id = utils.alpha2num(str(neighbors_of_edge_2[ids]))
                        region_dim = current_region_information.region_neighbor_dims[str(neighbors_of_edge_2[ids])]
                        if(num_region_id in undraped_region_list):
                            feature_val_undraped += region_dim
                        elif(num_region_id in draped_region_list):
                            feature_val_draped += region_dim



    total_perimeter = current_region_information.region_perimeter
    feature_val_undraped = feature_val_undraped/total_perimeter
    feature_val_draped = feature_val_draped/total_perimeter

    return [feature_val_undraped,feature_val_draped]



###Function to determine how far is the current region from draped region
def evaluate_region_proximity(region_information_dict, action_id, draped_region_list):
    feature_value = 0.0

    current_region_information = region_information_dict[str(action_id)]
    current_edgewise_neighbors = current_region_information.edgewise_region_neighbors

    computation_complete_flag = False
    for i in range(current_region_information.no_of_edges):
        current_edge_neighbor_ids = current_edgewise_neighbors[str(i)]
        internal_region_ids = np.where(current_edge_neighbor_ids !='O')[0]
        if(computation_complete_flag):
            break
        for neighbor in internal_region_ids:
            
            region_id = utils.alpha2num(current_edge_neighbor_ids[neighbor])
            
            ###If Immediate Neighbor is a draped region
            if(region_id in draped_region_list):
                feature_value = 0.0
                computation_complete_flag = True
                break
            
            ##If a corner neighbor is a draped region
            current_corner_neighbors = current_region_information.corner_neighbors
            
            corner_neighbor_ids = np.where(current_corner_neighbors != 'O')[0]
            loop_complete = False
            for corner_ids in corner_neighbor_ids:
                corner_id = utils.alpha2num(current_corner_neighbors[corner_ids])
                # print('Corner IDs: ', corner_id)
                if(corner_id in draped_region_list):
                    feature_value = 5.0
                    computation_complete_flag = True
                    loop_complete = True
                    break
            
            if(loop_complete):
                break
        
    if(not computation_complete_flag):
        feature_value = 15.0

    
    return feature_value


#####Function to calculate curvature interaction feature
def curvature_feature(region_information_dict, draped_region_list, action_id):
    feature_value = 0.0

    current_region_information = region_information_dict[str(action_id)]
    
    ## Concave Curvature(-5 -> -1) < Flat Curvature(1->5) < Convex Curvature(6->10)
    curvature_value = current_region_information.curvature
    if(len(draped_region_list) == 0):
        return curvature_value
    
    multiplication_factor = 1

    average_draped_region_curvature = 0.0
    for i in range(len(draped_region_list)):
        draped_region_id = utils.num2alpha(draped_region_list[i])
        current_draped_region_curvature = region_information_dict[str(draped_region_id)].curvature
        average_draped_region_curvature += current_draped_region_curvature

    
    average_draped_region_curvature = average_draped_region_curvature/(len(draped_region_list))
    
    change_in_curvature = np.abs(curvature_value - average_draped_region_curvature)
    feature_value = multiplication_factor*(change_in_curvature)

    # print('Curvature feature: ', feature_value)

    return feature_value


##Function to evaluate neighbor angle feature
def neighbor_angle_feature(region_information_dict, action_id, draped_region_list):

    feature_value = 0.0

    current_region_information = region_information_dict[str(action_id)]
    current_region_neighbors = current_region_information.edgewise_region_neighbors
    current_region_angles = current_region_information.angle_with_region

    for i in range(current_region_information.no_of_edges):
        current_edge_neighbors = current_region_neighbors[str(i)]

        for j in range(len(current_edge_neighbors)):
            if(current_edge_neighbors[j] != 'O'):
                current_neighbor_id = utils.alpha2num(current_edge_neighbors[j])
                current_angle = np.radians(current_region_angles[current_edge_neighbors[j]])

                if(current_neighbor_id in draped_region_list):
                    if(current_angle < 0 ):
                        feature_value += np.abs(current_angle)    ##Penalizing if doing current region was better
                    else:
                        feature_value -= current_angle            ##Rewarding if current region's number is draped with positive transtion cost

                else:
                    feature_value +=  current_angle
                
    # print('Feature Value: ', feature_value)
    feature_value = np.pi + feature_value
    # print('Neighbor Angles Feature Value: ', feature_value)

    return feature_value


####Defining a function to measure the shape complexity####
def shape_complexity_feature(region_information_dict, action_id):

    feature_value = 0.0
    current_region_information = region_information_dict[str(action_id)]
    feature_value = current_region_information.shape_complexity

    # print('Shape Complexity: ', feature_value)

    return feature_value


##Function to evaluate feature with z height
def z_height_feature(region_information_dict, action_id, draped_region_list):
    feature_value = 0.0
    current_region_information = region_information_dict[str(action_id)]
    max_height_component = (100/(20*current_region_information.z_max))
    if(len(draped_region_list) == 0):
        feature_value = max_height_component
        return feature_value

    z_height_avg = get_average_z_height(region_information_dict, draped_region_list)

    
    feature_value = z_height_avg - current_region_information.z_max
    if(feature_value<0):
        feature_value += 100*np.abs(feature_value)
    else:
        feature_value += 20*feature_value
    
    feature_value += max_height_component
    # print('Z Height feature: ', feature_value)
    return feature_value

####Function to get the highest z_max out of all the regions###
def get_heighest_zmax(region_information_dict, no_of_regions):

    z_max = 0.0
    region_id = 'a'
    for i in range(1,no_of_regions,1):
        current_region_id = utils.num2alpha(i)
        current_zmax = region_information_dict[current_region_id].z_max

        if(current_zmax > z_max):
            z_max = current_zmax
            region_id = current_region_id

    # print('Z max: ', z_max)
    # print('Region for Z max: ', region_id)

    return [z_max, region_id]

#########Function to compute the average z height of the draped regions###
#########################################################################
def get_average_z_height(region_information_dict, source_draped_regions):

    z_height_avg = 0.0

    if(len(source_draped_regions) == 0):
        return z_height_avg

    for i in range(len(source_draped_regions)):
        current_region_id = utils.num2alpha(source_draped_regions[i])
        z_height_avg += np.abs(region_information_dict[current_region_id].z_max)
    
    z_height_avg = z_height_avg/len(source_draped_regions)

    return z_height_avg


###Shape Convexity
###Inputs are region information dictionary, list of region or current state, flag whether to evaluate 
###for draped region or undraped region
##@This Feature needs to account for non uniform rows and columns later
##Right now this assumes that every row will have equal number of columns and vice versa#############################################
def evaluate_convexity(region_information_dict, state, draped_regions = True):

    convexity = 0.0
    no_of_regions = len(state)

    # print('Current State: ', state)
    source = np.zeros(no_of_regions, dtype=int)
    if(str(state) == str(source)):
        return convexity

    if(draped_regions):
        region_identifier = 1
    else:
        region_identifier = 0

    regions_of_interest = np.where(state == region_identifier)[0]
    regions_of_interest = regions_of_interest.astype(int) + 1

    # print('Regions of Interest: ', regions_of_interest)

    if(len(regions_of_interest) == 1):
        return convexity 


    region_indices_array = []
    area_of_regions_of_interest = 0.0
    for i in range(len(regions_of_interest)):
        region_id = utils.num2alpha(regions_of_interest[i])

        current_region_information = region_information_dict[str(region_id)]
        current_region_indices = current_region_information.region_index
        area_of_regions_of_interest += current_region_information.region_area

        if(i == 0):
            region_indices_array = np.array([current_region_indices]).astype(int)
        else:
            region_indices_array = np.append(region_indices_array, [current_region_indices], axis=0)

    min_indices = np.min(region_indices_array, axis=0)
    max_indices = np.max(region_indices_array, axis=0)

    min_row = min_indices[0]
    max_row = max_indices[0]

    min_col = min_indices[1]
    max_col = max_indices[1]


    
    area_of_bounding_box = 0.0 

    for i in range(1,no_of_regions+1,1):
        region_id = utils.num2alpha(i)

        current_region_information = region_information_dict[str(region_id)]
        current_region_indices = current_region_information.region_index

        if((min_row <= current_region_indices[0] <= max_row) and (min_col <= current_region_indices[1] <= max_col)):
            # print(region_id, ' Region Area: ', current_region_information.region_area)
            area_of_bounding_box += current_region_information.region_area
           
    convexity = 1 - (area_of_regions_of_interest/area_of_bounding_box)

    # print('Value of Convexity: ', convexity)

    return convexity

#############################Tiling Problem############################################

#####Feature Evaluation function for tiling problem
def evaluate_tile_features(source, destination, tile_features_dict, window_dimensions,number_of_features = 4):

    feature_vector = np.zeros(number_of_features)
    no_of_tiles = len(source)
    ##Converting Source Vector to approriate one hot encoding
    source_vec = copy.deepcopy(np.array(source))
    source_vec = source_vec.astype(int)
    
    ##Getting the Visited and Unvisited tiles in Source State
    source_visited_tiles = np.where(source_vec == 1)[0]
    source_visited_tiles = source_visited_tiles + 1
    source_visited_tiles = source_visited_tiles.astype(int)

    source_unvisited_tiles = np.where(source_vec == 0)[0]
    source_unvisited_tiles = source_unvisited_tiles + 1
    source_unvisited_tiles = source_unvisited_tiles.astype(int)

    ##Converting Destination Vector to approriate one hot encoding
    destination_vec = copy.deepcopy(np.array(destination))
    destination_vec = destination_vec.astype(int)
    
    ##Getting the Visited and Unvisited tiles in Destination State
    destination_visited_tiles = np.where(destination_vec == 1)[0]
    destination_visited_tiles = destination_visited_tiles + 1
    destination_visited_tiles = destination_visited_tiles.astype(int)

    destination_unvisited_regions = np.where(destination_vec == 0)[0]
    destination_unvisited_regions = destination_unvisited_regions + 1
    destination_unvisited_regions = destination_unvisited_regions.astype(int)
    
    
    action_vec = destination_vec - source_vec

    action = np.where(action_vec == 1)[0]
    action = int(action[0]) + 1
    action_id = utils.num2alpha(action)
    # print("Action ID: ", action_id)
    tile_information_dict = copy.deepcopy(tile_features_dict)


    #####Evaluating individual features#####

    ###First feature is computing convexity of the visited tiles in the source
    convexity_feature = evaluate_tile_convexity(tile_information_dict, source, action_id)


    ###Second feature is computing the max distance from the leftmost edge for the current action
    length_of_left_empty_space = evaluate_empty_spaces(tile_information_dict, source, action_id, window_dimensions,4)

    ###Third feature is computing the max distance from the topmost visited edge for the current action
    length_of_empty_spaces_above = evaluate_empty_spaces(tile_information_dict, source, action_id, window_dimensions,1)


    ###Fourth Feature is computing the max distance above that is left unvisited
    length_of_max_empty_space_above = evaluate_max_empty_spaces(tile_information_dict, source, destination, action_id, window_dimensions,1)

    feature_vector = [convexity_feature, length_of_left_empty_space, length_of_empty_spaces_above, length_of_max_empty_space_above]
    
    return feature_vector

####Tile convexity feature#####
def evaluate_tile_convexity(tile_information_dict, current_state, current_action, compute_for_visited = True):

    convexity_feature = 0.0

    ##compute the area of visited regions
    ##compute the area of bounding box
    no_of_regions = len(current_state)

    # print('Current State: ', state)
    source = np.zeros(no_of_regions, dtype=int)
    if(str(current_state) == str(source)):
        return convexity_feature

    if(compute_for_visited):
        region_identifier = 1
    else:
        region_identifier = 0

    tiles_of_interest = np.where(current_state == region_identifier)[0]
    tiles_of_interest = tiles_of_interest.astype(int) + 1

    # print('Tiles of Interest: ', tiles_of_interest)

    if(len(tiles_of_interest) == 1):
        return convexity_feature 


    area_of_tile_of_interest = 0.0
    
    for i in range(len(tiles_of_interest)):
        tile_id = utils.num2alpha(tiles_of_interest[i])

        current_tile_information = tile_information_dict[str(tile_id)]
        
        area_of_tile_of_interest += current_tile_information.tile_area

        current_tile_corner_coords = current_tile_information.corner_coords


        if(i == 0):
            tile_corner_array = np.array(current_tile_corner_coords)
        else:
            tile_corner_array = np.append(tile_corner_array, current_tile_corner_coords, axis=0)

    # print(tile_corner_array)
    min_indices = np.min(tile_corner_array, axis=0)
    max_indices = np.max(tile_corner_array, axis=0)

    # print('Min Indices: ', min_indices)
    # print('Max Indices: ', max_indices)

    min_x = min_indices[0]
    max_x = max_indices[0]

    min_y = min_indices[1]
    max_y = max_indices[1]

    area_of_bounding_box = (max_x-min_x) * (max_y-min_y)

    convexity_feature = 1 - (area_of_tile_of_interest/area_of_bounding_box)
    # print('Tile Corber Array: ', tile_corner_array)
    # print("Convexity Feature for Tile: ", convexity_feature)

    return convexity_feature


###Computing Empty Space to the left###
def evaluate_empty_spaces(tile_information_dict, source, action_id, window_dimensions, edge_id):

    empty_space_length = 0.0

    dimension_id = 0
    if(edge_id == 1 or edge_id==3):
        dimension_id = 1
    
    tile_id = action_id
    
    ##Get the tiles to the left
    tiles_of_interest = tile_information_dict[tile_id].edgewise_tile_neighbors[str(edge_id)]

    ##If there are no 
    if(len(tiles_of_interest) == 0):
        return empty_space_length

    # print('Tiles of interest: ', tiles_of_interest)
    tile_accounted_for = np.zeros(len(source))

    for i in range(len(tiles_of_interest)):
        current_set_of_tiles = tiles_of_interest[i]
        
        break_flag = True
        current_set_of_unvisited_tiles_lengths = []
        for j in range(len(current_set_of_tiles)):
            current_tile = current_set_of_tiles[j]
            
            current_tile_information = tile_information_dict[current_tile]
            current_tile_dimensions = current_tile_information.tile_dims
            
            ##Check if current tile is visited i.e. source[current_tile] = 1
            current_tile_id = utils.alpha2num(current_tile)
            
            if(source[current_tile_id-1] == 0 and tile_accounted_for[current_tile_id-1] == 0):
                break_flag = False
                current_set_of_unvisited_tiles_lengths.append(current_tile_dimensions[dimension_id])
                tile_accounted_for[current_tile_id-1] = 1

        # print(current_set_of_tiles_lengths)
        
        if(len(tiles_of_interest)==1 and len(current_set_of_unvisited_tiles_lengths) != 0):
            empty_space_length = np.sum(current_set_of_unvisited_tiles_lengths)
            break
        
        if(len(current_set_of_unvisited_tiles_lengths) != 0):     
            empty_space_length += np.max(current_set_of_unvisited_tiles_lengths)
        
        if(break_flag):
            # print("Exited from the loop")
            break
    
    normalization_factor = window_dimensions[dimension_id]
   
    empty_space_length = empty_space_length/normalization_factor
    # print("Empty Space to edge", edge_id, "feature: ", empty_space_length)

    return empty_space_length


##This function is used to compute max length of the empty spaces above or below
def evaluate_max_empty_spaces(tile_information_dict, source, destination, action_id, window_dimensions,edge_id):

    length_of_max_empty_space = 0.0

    ##Find the y_min for current region
    current_action_info = tile_information_dict[action_id]
    current_action_corner_coords = current_action_info.corner_coords
    current_action_min_y = np.min(current_action_corner_coords, axis=0)[1]
    
    if(current_action_min_y == 0):
        return length_of_max_empty_space

    
    unvisited_tiles = np.where(destination == 0)[0]
    unvisited_tiles = unvisited_tiles + 1
    unvisited_tiles = unvisited_tiles.astype(int)
    # print("Unvisited TIles: ", unvisited_tiles)
    first_level_unvisited_tiles = []
    
    for i in range(len(unvisited_tiles)):
        current_tile_id = utils.num2alpha(unvisited_tiles[i])
        current_tile_info = copy.deepcopy(tile_information_dict[current_tile_id])
        current_tile_corner_coords = current_tile_info.corner_coords
        min_indices = np.min(current_tile_corner_coords, axis=0)
        current_ymin = min_indices[1]

        if(current_ymin == 0):
            first_level_unvisited_tiles.append(current_tile_id)

    if(len(first_level_unvisited_tiles) == 0):
        return length_of_max_empty_space

    empty_space_array = []
    for i in range(len(first_level_unvisited_tiles)):
        current_tile_id = first_level_unvisited_tiles[i]
        current_tile_info = copy.deepcopy(tile_information_dict[current_tile_id])
        current_tile_empty_space = evaluate_empty_spaces(tile_information_dict, destination, action_id, window_dimensions,1)
        # print("Value here: ", current_tile_empty_space)
        if(current_tile_empty_space>current_action_min_y):
            current_tile_empty_space = copy.deepcopy(current_action_min_y)
        
        empty_space_array.append(current_tile_empty_space)

    
    length_of_max_empty_space = np.max(empty_space_array)
    

    return length_of_max_empty_space


###########Synthetic Data Features##########################

def evaluate_synthetic_features(source,destination, region_features_dict, mold_dimension, number_of_features):

    feature_vector = np.zeros(number_of_features)
    no_of_regions = len(source)
    ##Converting Source Vector to approriate one hot encoding
    source_vec = copy.deepcopy(np.array(source))
    source_vec = source_vec.astype(int)
    ##Getting the Draped and Undraped regions in Source State
    source_draped_regions = np.where(source_vec == 1)[0]
    source_draped_regions = source_draped_regions + 1
    source_draped_regions = source_draped_regions.astype(int)

    source_undraped_regions = np.where(source_vec == 0)[0]
    source_undraped_regions = source_undraped_regions + 1
    source_undraped_regions = source_undraped_regions.astype(int)

    ##Converting Destination Vector to approriate one hot encoding
    destination_vec = copy.deepcopy(np.array(destination))
    destination_vec = destination_vec.astype(int)
    ##Getting the Draped and Undraped regions in Destination State
    destination_draped_regions = np.where(destination_vec == 1)[0]
    destination_draped_regions = destination_draped_regions + 1
    destination_draped_regions = destination_draped_regions.astype(int)

    destination_undraped_regions = np.where(destination_vec == 0)[0]
    destination_undraped_regions = destination_undraped_regions + 1
    destination_undraped_regions = destination_undraped_regions.astype(int)
    
    
    action_vec = destination_vec - source_vec

    action = np.where(action_vec == 1)[0]
    action = int(action[0]) + 1
    action_id = utils.num2alpha(action)
    # print("Action ID: ", action_id)
    region_information_dict = copy.deepcopy(region_features_dict)


    ##Feature 1
    feature_1 = evaluate_feature_internal_free_synthetic(region_information_dict, source_undraped_regions, action_id)

    ##Feature 2
    feature_2 = evaluate_feature_internal_constrained_synthetic(region_information_dict, source_draped_regions, action_id)

    ##Feature 3 and 4
    [feature_3, feature_4] = evaluate_boundary_opp_status_synthetic(region_information_dict, action_id, source_undraped_regions, source_draped_regions)

    #Feature 5
    feature_5 = evaluate_region_proximity_synthetic(region_information_dict, action_id, source_draped_regions)

    #Feature 6
    feature_6 = evaluate_convexity_synthetic(region_information_dict, source, True)

    #Feature 7
    feature_7 = evaluate_convexity_synthetic(region_information_dict, source, True)

    #Feature 8
    feature_8 = curvature_feature_synthetic(region_information_dict, source_draped_regions, action_id)

    #Feature 9
    feature_9 = neighbor_angle_feature_synthetic(region_information_dict, action_id, source_draped_regions)
    # feature_9 = region_features_dict[str(action_id)].region_angle

    #Feature 10
    feature_10 = z_height_feature_synthetic(region_information_dict, action_id, source_draped_regions)

    feature_vector = [feature_1, feature_2, feature_3, feature_4, feature_5,feature_6,feature_7, feature_8, feature_9, feature_10]
    
    return feature_vector


##Feature is counting the number of internal edges that are undraped
def evaluate_feature_internal_free_synthetic(region_information_dict, undraped_region_list,region_id):

    ##Length of Undraped Internal Edges
    feature_value = 0.0
    
    current_region_information = region_information_dict[str(region_id)]
    current_edgewise_neighbors = current_region_information.edgewise_region_dims
    
    for i in range(current_region_information.no_of_edges):
        neighbor_id = current_edgewise_neighbors[str(i)]

        for neighbor in neighbor_id:
            neighbor_num_id = utils.alpha2num(neighbor)
            if(neighbor != 'O' and (neighbor_num_id in undraped_region_list)):
                region_dim = current_region_information.region_neighbor_dims[str(neighbor)]
                feature_value += region_dim
 
    perimeter_of_region = 2*(current_region_information.region_dims[0] + current_region_information.region_dims[1])
    
    feature_value = feature_value/perimeter_of_region

    
    return feature_value


##Feature is counting the number of internal edges that are undraped
def evaluate_feature_internal_constrained_synthetic(region_information_dict, draped_region_list, region_id):

    ##Length of Undraped Internal Edges
    feature_value = 0.0
 
    current_region_information = region_information_dict[str(region_id)]
    current_edgewise_neighbors = current_region_information.edgewise_region_dims
    
    for i in range(current_region_information.no_of_edges):
        neighbor_ids = current_edgewise_neighbors[str(i)]

        for neighbor in neighbor_ids:
            neighbor_num_id = utils.alpha2num(neighbor)
            
            if(neighbor != 'O' and (neighbor_num_id in draped_region_list)):
                region_dim = current_region_information.region_neighbor_dims[str(neighbor)]    
                feature_value += region_dim            
    
    perimeter_of_region = 2*(current_region_information.region_dims[0] + current_region_information.region_dims[1])
    feature_value = feature_value/perimeter_of_region
    # print('Value of Feature 2: ', feature_value)
    return feature_value

###Function to evaluate the edges opposite a boundary edge being undraped or draped
###This implementation assumes that every region will have exactly 4 edges
def evaluate_boundary_opp_status_synthetic(region_information_dict, action_id, undraped_region_list, draped_region_list):

    ###Length of the boundary edges for which opposite region is undraped
    feature_val_undraped = 0.0

    ###Length of the boundary edges for which opposite region is draped
    feature_val_draped = 0.0
    
    current_region_information = region_information_dict[str(action_id)]
    current_edgewise_neighbors = current_region_information.edgewise_region_dims

    for i in range(current_region_information.no_of_edges):
        current_edge_neighbor_ids = current_edgewise_neighbors[str(i)]

        for neighbor in current_edge_neighbor_ids:
            
            if(neighbor == 'O'):
                #Along first edge
                if(i == 0):
                    neighbors_of_edge_3 = current_region_information.edgewise_region_dims[str(2)]
                    internal_region_ids = np.where(neighbors_of_edge_3 !='O')[0]
                  
                    # print("i = 0: ", internal_region_ids)
                    for ids in internal_region_ids:
                        num_region_id = utils.alpha2num(str(neighbors_of_edge_3[ids]))
                        region_dim = current_region_information.region_neighbor_dims[str(neighbors_of_edge_3[ids])]
                        if(num_region_id in undraped_region_list):
                            feature_val_undraped += region_dim
                        elif(num_region_id in draped_region_list):
                            feature_val_draped += region_dim

                elif(i == 1):
                    neighbors_of_edge_4 = current_region_information.edgewise_region_dims[str(3)]
                    internal_region_ids = np.where(neighbors_of_edge_4 !='O')[0]
                    
                    # print("i = 1", internal_region_ids)
                    for ids in internal_region_ids:
                        num_region_id = utils.alpha2num(str(neighbors_of_edge_4[ids]))
                        region_dim = current_region_information.region_neighbor_dims[str(neighbors_of_edge_4[ids])]
                        if(num_region_id in undraped_region_list):
                            feature_val_undraped += region_dim
                        elif(num_region_id in draped_region_list):
                            feature_val_draped += region_dim

                elif(i == 2):
                    neighbors_of_edge_1 = current_region_information.edgewise_region_dims[str(0)]
                    internal_region_ids = np.where(neighbors_of_edge_1 !='O')[0]
                    # print("i = 2: ", internal_region_ids)
                    for ids in internal_region_ids:
                        num_region_id = utils.alpha2num(str(neighbors_of_edge_1[ids]))
                        region_dim = current_region_information.region_neighbor_dims[str(neighbors_of_edge_1[ids])]
                        if(num_region_id in undraped_region_list):
                            feature_val_undraped += region_dim
                        elif(num_region_id in draped_region_list):
                            feature_val_draped += region_dim
                
                elif(i == 3):
                    neighbors_of_edge_2 = current_region_information.edgewise_region_dims[str(1)]
                    internal_region_ids = np.where(neighbors_of_edge_2 !='O')[0]
                    # print("i = 3: ", internal_region_ids)
                    for ids in internal_region_ids:
                        num_region_id = utils.alpha2num(str(neighbors_of_edge_2[ids]))
                        region_dim = current_region_information.region_neighbor_dims[str(neighbors_of_edge_2[ids])]
                        if(num_region_id in undraped_region_list):
                            feature_val_undraped += region_dim
                        elif(num_region_id in draped_region_list):
                            feature_val_draped += region_dim



  
    total_perimeter = current_region_information.region_dims[0] + current_region_information.region_dims[1]

    feature_val_undraped = feature_val_undraped/total_perimeter
    feature_val_draped = feature_val_draped/total_perimeter

    
    return [feature_val_undraped,feature_val_draped]


##Feature 5 is the distance between currently draped region and the already draped region
def evaluate_region_proximity_synthetic(region_information_dict, action_id, draped_region_list):

    feature_value = 0.0

    current_region_information = region_information_dict[str(action_id)]
    current_edgewise_neighbors = current_region_information.edgewise_region_dims
    
    if(len(draped_region_list) == 0):
        # print("feature value from source: ", feature_value)
        return feature_value

   

    computation_complete_flag = False
    for i in range(current_region_information.no_of_edges):
        current_edge_neighbor_ids = current_edgewise_neighbors[str(i)]
        internal_region_ids = np.where(current_edge_neighbor_ids !='O')[0]
        if(computation_complete_flag):
            break
        for neighbor in internal_region_ids:
    
            region_id = utils.alpha2num(current_edge_neighbor_ids[neighbor])
            
            ###If Immediate Neighbor is a draped region
            if(region_id in draped_region_list):
                feature_value = 0.0
                computation_complete_flag = True
                return feature_value
            
            
 
    ##If a corner neighbor is a draped region
    current_corner_neighbors = current_region_information.corner_neighbors
    
    corner_neighbor_ids = np.where(current_corner_neighbors != 'O')[0]
    for corner_ids in corner_neighbor_ids:
        corner_id = utils.alpha2num(current_corner_neighbors[corner_ids])
        # print('Corner IDs: ', corner_id)
        if(corner_id in draped_region_list):
            feature_value = 5.0
            computation_complete_flag = True
            break

    if(not computation_complete_flag):
        feature_value = 15.0

    
    return feature_value


def evaluate_convexity_synthetic(region_information_dict, current_state, compute_for_visited = True):

    convexity_feature = 0.0

    ##compute the area of visited regions
    ##compute the area of bounding box
    no_of_regions = len(current_state)

    source = np.zeros(no_of_regions, dtype=int)
    if(str(current_state) == str(source)):
        return convexity_feature

    if(compute_for_visited):
        region_identifier = 1
    else:
        region_identifier = 0

    regions_of_interest = np.where(current_state == region_identifier)[0]
    regions_of_interest = regions_of_interest.astype(int) + 1

    
    if(len(regions_of_interest) == 1):
        return convexity_feature 


    area_of_region_of_interest = 0.0
    
    for i in range(len(regions_of_interest)):
        region_id = utils.num2alpha(regions_of_interest[i])

        current_tile_information = region_information_dict[str(region_id)]
        
        area_of_region_of_interest += current_tile_information.region_area

        current_region_corner_coords = current_tile_information.corner_coords


        if(i == 0):
            region_corner_array = np.array(current_region_corner_coords)
        else:
            region_corner_array = np.append(region_corner_array, current_region_corner_coords, axis=0)

    min_indices = np.min(region_corner_array, axis=0)
    max_indices = np.max(region_corner_array, axis=0)

    
    min_x = min_indices[0]
    max_x = max_indices[0]

    min_y = min_indices[1]
    max_y = max_indices[1]

    area_of_bounding_box = (max_x-min_x) * (max_y-min_y)

    convexity_feature = 1 - (area_of_region_of_interest/area_of_bounding_box)
    
    return convexity_feature


def curvature_feature_synthetic(region_information_dict, draped_region_list, action_id):
    feature_value = 0.0

    current_region_information = region_information_dict[str(action_id)]
    
    ## Concave Curvature(-5 -> -1) < Flat Curvature(1->5) < Convex Curvature(6->10)
    curvature_value = current_region_information.curvature
    
    #Preferring to do flat curvature regions first
    if(len(draped_region_list) == 0):
        curvature_value = np.abs(curvature_value)
        return curvature_value

    average_draped_region_curvature = 0.0
    for i in range(len(draped_region_list)):
        draped_region_id = utils.num2alpha(draped_region_list[i])
        current_draped_region_curvature = region_information_dict[str(draped_region_id)].curvature
        average_draped_region_curvature += current_draped_region_curvature

    
    average_draped_region_curvature = average_draped_region_curvature/(len(draped_region_list))
    
    change_in_curvature = curvature_value - average_draped_region_curvature

    if(change_in_curvature<0):
        feature_value = 0
    else:
        feature_value = change_in_curvature
    
    
    return feature_value

def neighbor_angle_feature_synthetic(region_information_dict, action_id, draped_region_list):

    feature_value = 0.0

    current_region_information = region_information_dict[str(action_id)]
    current_region_neighbors = current_region_information.edgewise_region_dims
    current_region_angle = current_region_information.region_angle

    for i in range(current_region_information.no_of_edges):
        current_edge_neighbors = current_region_neighbors[str(i)]

        for j in range(len(current_edge_neighbors)):
            if(current_edge_neighbors[j] != 'O'):
                current_neighbor_id = utils.alpha2num(current_edge_neighbors[j])
                
                if(current_neighbor_id not in draped_region_list):
                    current_neighbor_angle = region_information_dict[str(current_edge_neighbors[j])].region_angle
                    if(current_region_angle != current_neighbor_angle):
                        feature_value += (1/np.abs(current_region_angle - current_neighbor_angle))
                
                
    return feature_value

def z_height_feature_synthetic(region_information_dict, action_id, draped_region_list):
    feature_value = 0.0
    current_region_information = region_information_dict[str(action_id)]
    max_height_component = current_region_information.z_max
    
    if(len(draped_region_list) == 0):
        if(max_height_component == 0):
            return max_height_component
        feature_value = np.abs(1/max_height_component)
        return feature_value

    z_height_avg = get_average_z_height(region_information_dict, draped_region_list)

    
    average_height_component = z_height_avg - current_region_information.z_max
    
    if(average_height_component<0):
        average_height_component = 3*(np.abs(average_height_component))

       
    if(max_height_component == 0):
        feature_value = 0
    else:
        feature_value = 1/max_height_component
    
    feature_value += average_height_component
    
    feature_value = feature_value/2
    
    # print('Z Height feature: ', feature_value)
    
    return feature_value