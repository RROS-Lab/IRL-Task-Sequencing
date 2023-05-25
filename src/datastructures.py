###########This File Consists of Two Datastructures: A node and corresponding Graph##########
#######Maintainer: Omey Mohan Manyar, Email ID: manyar@usc.edu##################################


####Imports######
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from queue import PriorityQueue
import copy
import re
import feature_utilities
import utilities as utils
import timeit
#################

###A Node in the Corresponding Graph is Composed of an encoding of Visited and Unvisited Regions#####
###This also stores the information of the edge weights and children ids#############
###The feature vector for each of the edges is stored in this datastructure as well#####
class Node:

    def __init__(self,state_id):
        self.id = state_id
        self.children = {}
        self.child_features = {}
        return


    def add_child(self, child_vertex, weight):
        self.children[str(child_vertex)] = weight
        return
    
    def define_child_features(self, child_vertex, feature_vector):
        self.child_features[str(child_vertex)] = feature_vector
        return

    def get_child_features(self, child_vertex):
        return self.child_features[str(child_vertex)]

    def get_children_ids(self):
        return self.children.keys()


    def remove_child(self, child_vertex):
        self.children.pop(str(child_vertex), None)
        self.child_features.pop(str(child_vertex), None)

        return

    def get_Node_id(self):
        return self.id
    
    def get_edge_weight(self, child_id):
        return self.children[str(child_id)]


###This class stores the information of geometrical features for a particular region/grid member#####
###Examples of the features are: Curvature, region neighbors, relative_angles, shape complexity, etc.
class region_features:

    def __init__(self, region_id):
        
        ##ID for the region
        self.region_id = region_id
        self.region_no = 1

        ##No of edges Default is 4 since default is rectangle or square
        self.no_of_edges = 4
        
        ##This dict containts keys: 0,1,2,3 and the corresponding neighbors for each of them
        self.edgewise_region_neighbors = {}
        
        ##This dict containts keys as the region neighbor ids and the length they share
        self.region_neighbor_dims = {}

        ##This dict containts keys as 0,1,2,3 as edges and corresponding boundary region dimensions
        self.edgewise_boundary_dims = {}
        
        ###Angles with the neighboring regions
        ###Key is the corresponding region id
        self.angle_with_region = {}

        #####Corner Neighbors, these are the regions that are on the corners of the given vertex 
        self.corner_neighbors = []

        ###Curvature of the region
        self.curvature = 0.0

        ###Shape Complexity
        self.shape_complexity = 0.0

        ##Region Area
        self.region_area = 0.0

        ##Average z height
        self.z_avg = 0.0

        ##Max Z height
        self.z_max = 0.0

        ##Region Index [row, col]
        self.region_index = []


        ##Region Perimeter
        self.region_perimeter = 0

        return

######Synthetic Data Region Features####
class synth_region_features:

    def __init__(self, region_id):
        
        ##ID for the region
        self.region_id = region_id
        self.region_no = 1

        ##No of edges Default is 4 since default is rectangle or square
        self.no_of_edges = 4
        
        ##This dict containts keys: 0,1,2,3 and the corresponding neighbors for each of them
        self.edgewise_region_neighbors = {}
        
        ##This dict containts keys as the region neighbor ids and the length they share
        self.region_neighbor_dims = {}

        ##This dict containts keys as 0,1,2,3 as edges and corresponding boundary region dimensions
        self.edgewise_region_dims = {}
        
        ###Angles with the neighboring regions
        ###Key is the corresponding region id
        self.angle_with_region = {}

        #####Corner Neighbors, these are the regions that are on the corners of the given vertex 
        self.corner_neighbors = []

        ####Corner Coordinates
        self.corner_coords = []

        ###Curvature of the region
        self.curvature = 0.0

        ##Region Area
        self.region_area = 0.0

        ##Region Angle
        self.region_angle = 0.0

        ##Max Z height
        self.z_max = 0.0

        ###Region Dimensions
        self.region_dims = []


        ##Region centroid
        self.region_centroid = []       ##This is the (x,y) coordinate of the region centroid

    
        return


##This is a class implementation for the tile class
class tile:

    ###Constructor for the tile datastructure
    def __init__(self, tile_id):

        ##ID for the region
        self.tile_id = tile_id
        self.tile_no = 1

        ##No of edges Default is 4 since default is rectangle or square
        self.no_of_edges = 4
        
        ##This dict containts keys: 0,1,2,3 and the corresponding neighbors for each of them
        self.edgewise_tile_neighbors = {}
        
        ##This dict containts keys as the region neighbor ids and the length they share
        self.tile_neighbor_dims = {}

        ##This dict containts keys as 0,1,2,3 as edges and corresponding boundary region dimensions
        self.edgewise_boundary_dims = {}

        #####Corner Neighbors, these are the regions that are on the corners of the given vertex 
        self.corner_neighbors = []


        ###Corner Coordinates
        self.corner_coords = []

        #####Tile Dimensions
        self.tile_dims = []

        ##Region Area
        self.tile_area = 0.0


        ##Region position
        self.tile_position = {}         ##Dictionary having keys 0 -> regions above, 1 -> regions right, 3 -> regions below, 4-> regions left

        

        return

#######Graph to store the information of Nodes and edges########
#######For the specific example this graph represents the mold, the nodes are the regions######
class Graph:
    
    ####Constructor for the graph datastructure####
    def __init__(self, filename, graph_type = "real"):
        
        ###Flag for tile vs mold problem
        self.graph_id = graph_type
        ##Dictionary to store region feature information
        ##Each key is a,b,c,..... region id and the value is an object of the class region_features
        self.region_information_dict = {}

        ####User Preferred Path#######
        self.user_preferred_path_alpha = []
        self.user_preferred_path_num = []

        #Defining graph and edge features
        self.feature_dict = {}
        if(self.graph_id == "tile"):
            self.no_of_features = 4                 #Using 4 features for the tile problem
        else:
            self.no_of_features = 10                #Using 10 features for the mold composite problem

        # self.features_weights = np.random.rand(self.no_of_features)        
        self.features_weights = np.ones(self.no_of_features)
        # self.features_weights = [2,1,4,8,12,18,128,25,55,50]
        # self.features_weights = [8,4,2,3]
        norm_of_weights = np.linalg.norm(self.features_weights)
        self.features_weights = self.features_weights/norm_of_weights
        # self.features_weights = [0.58837753,0.55300278,0.43834525,0.39478253]
        ##Initializing the dictionary for region information
        if(self.graph_id == "tile"):
            self.parse_graph_tile(filename)
        elif(self.graph_id == "real"):
            self.parse_graph(filename)
        elif(self.graph_id == "synthetic"):
            self.parse_graph_synthetic(filename)


        ##Defining the Dictionary to hold all the nodes/states in the graph
        self.vert_dict = {}
        self.state_ids = []
        self.max_vertices = np.power(2, self.no_of_regions)
        self.num_vertices = 0
        

        # #Adding the source node to the graph
        self.source_node = np.zeros(self.no_of_regions, dtype=int)
        self.goal_node = np.ones(self.no_of_regions, dtype=int)
        
        ###Generating all nodes/states
        self.state_id_vector = [None] * self.no_of_regions
        self.graph_initialized = False
        
        print('Generating State Space')
        self.generate_state_space(self.no_of_regions)
        print('Completed generating state space')
        self.state_ids = np.reshape(self.state_ids, [self.max_vertices, self.no_of_regions])
        self.state_ids = self.state_ids.astype(int)
        self.state_ids = np.unique(self.state_ids, axis=0)

        self.initialize_graph()
        print("Graph Initialization Completed")

        return


    ###Reads the csv with name: filename
    ### This should be stored in the folder data within the workspace
    def parse_graph(self, filename):

        self.max_vertices = 1
        self.no_of_regions = 1

        ###Reading the CSV information
        [fields, rows] = utils.read_csv(filename)
        # print('Fields: ', fields)
        # print('Rows: ', rows)
        

        ##No of Regions in the given graph
        self.no_of_regions = len(rows)
        print('No of regions: ', self.no_of_regions)

        ####Defining the User Preferred Path
        user_preffered_path = rows[0][11]
        user_preffered_path = user_preffered_path.split(",")
        user_preffered_path = np.array(user_preffered_path).astype(str)

        self.user_preferred_path_alpha = user_preffered_path
        self.user_preferred_path_num = utils.convert_alpha_path_to_num_path(user_preffered_path)
        
        # print('Alpha Path: ', self.user_preferred_path_alpha)
        # print('Num Path: ', self.user_preferred_path_num)
        # print('Function test num path: ', utils.convert_num_path_to_alpha_path(self.user_preferred_path_num))
        
        ###Reading the region information row by row in a specific order
        ###The data format is important here and pretty rigid, refer to the csv's in data folder
        for i in range(len(rows)):
            current_row = rows[i]
            ##Getting the region no and region id
            region_no = i+1
            region_id = current_row[0]
            current_region_features = region_features(region_id)

            ##Initializing region index
            region_index = current_row[10]
            region_index = region_index.split(",")
            region_index = np.array(region_index).astype(int)
            current_region_features.region_index = region_index

            # print('Region index for region ', current_region_features.region_id, ' is ', current_region_features.region_index)

            ##Region Number, order in which the number occurs
            current_region_features.region_no = region_no
            # print("Region No is: ", current_region_features.region_no)
            
            ##Getting the curvature
            curvature = float(current_row[1])
            # print('Curvature: ', curvature)
            current_region_features.curvature = copy.deepcopy(curvature)

            ###Region Neighbors
            region_neighbors = current_row[2]
            region_neighbors = region_neighbors.split(",")
            region_neighbors = np.array(region_neighbors).astype(str)
            split_indices = np.where(region_neighbors == 'x')[0]
            region_neighbors = np.delete(region_neighbors, split_indices)
            split_indices = split_indices - np.arange(0, len(split_indices))
            edgewise_neighbors = np.split(region_neighbors, split_indices)
            # print('Region Neighbors: ', edgewise_neighbors)

            ##Angles with region
            region_angles = current_row[3]
            region_angles = region_angles.split(",")
            region_angles = np.array(region_angles).astype(str)
            # print('Region Angles: ', region_angles)

            ###Region Lengths
            region_lengths = current_row[4]
            region_lengths = region_lengths.split(",")
            region_lengths = np.asfarray(region_lengths)
            # print('Region Lengths: ', region_lengths)
            #Region Perimeter
            current_region_features.region_perimeter = np.sum(region_lengths)

            ##No of edges for the given region
            current_region_features.no_of_edges = len(edgewise_neighbors)
            # print("No of edges: ", current_region_features.no_of_edges)
            
            start_index = 0

            for j in range(len(edgewise_neighbors)):
                current_edge_neighbors = edgewise_neighbors[j]
                current_edge_neighbors_dims = region_lengths[start_index: start_index + len(current_edge_neighbors)]
                
                current_region_features.edgewise_region_neighbors[str(j)] = edgewise_neighbors[j]

                for k in range(len(current_edge_neighbors)):
                    
                    if(current_edge_neighbors[k] == 'O'):
                        if(str(j) in current_region_features.edgewise_boundary_dims.keys()):
                            current_boundary_length = current_region_features.edgewise_boundary_dims[str(j)]
                            current_region_features.edgewise_boundary_dims[str(j)] = current_boundary_length + current_edge_neighbors_dims[k]
                        else:
                            current_region_features.edgewise_boundary_dims[str(j)] = current_edge_neighbors_dims[k]
                        
                    else:
                        # print('Current Region: ', current_region_features.region_id)
                        # print(' Neighbor: ', current_edge_neighbors[k])
                        # print(' Region Angles: ', region_angles)
                        # print('start index: ', start_index)
                        # print('k: ', k)
                      
                        current_region_features.angle_with_region[str(current_edge_neighbors[k])] = float(region_angles[start_index+k])
                        current_region_features.region_neighbor_dims[str(current_edge_neighbors[k])] = current_edge_neighbors_dims[k]
                        

                start_index += len(edgewise_neighbors[j]) 
            

            # print('Current Region Boundaries', current_region_features.edgewise_boundary_dims)
            # print('Neighbor Region Dims', current_region_features.region_neighbor_dims)
            # print('Angles with region: ', current_region_features.angle_with_region)

            ##Shape Complexity
            current_region_features.shape_complexity = float(current_row[5])
            # print('Shape Complexity: ', current_region_features.shape_complexity)

            ##Area
            current_region_features.region_area = float(current_row[6])
            # print('Region Area: ', current_region_features.region_area)

            ##Z Average
            current_region_features.z_avg = float(current_row[7])
            # print('Z Avg: ', current_region_features.z_avg)

            ##Z Max
            current_region_features.z_max = float(current_row[8])
            # print('Z Max: ', current_region_features.z_max)


            ###Corner Neighbors
            corner_neighbors = current_row[9]
            corner_neighbors = corner_neighbors.split(",")
            corner_neighbors = np.array(corner_neighbors).astype(str)
            current_region_features.corner_neighbors = corner_neighbors

            self.region_information_dict[str(region_id)] = copy.deepcopy(current_region_features)
            


        # print('Region Information Dictionary: ', self.region_information_dict)

        return


    def parse_graph_synthetic(self, filename):

        self.max_vertices = 1
        self.no_of_regions = 1

        ###Reading the CSV information
        [fields, rows] = utils.read_csv(filename)
        # print('Fields: ', fields)
        # print('Rows: ', rows)
        

        ##No of Regions in the given graph
        self.no_of_regions = len(rows)
        # print('No of regions: ', self.no_of_regions)

        ####Defining the User Preferred Path
        user_preffered_path = rows[0][10]
        user_preffered_path = user_preffered_path.split(",")
        user_preffered_path = np.array(user_preffered_path).astype(str)

        self.user_preferred_path_alpha = user_preffered_path
        self.user_preferred_path_num = utils.convert_alpha_path_to_num_path(user_preffered_path)
        
        # print('Alpha Path: ', self.user_preferred_path_alpha)
        # print('Num Path: ', self.user_preferred_path_num)
        # print('Function test num path: ', utils.convert_num_path_to_alpha_path(self.user_preferred_path_num))
        
        ##Total Mold Dimensions
        mold_dims = rows[0][9]
        mold_dims = mold_dims.split(",")
        # print(window_dims)
        mold_dims = np.asfarray(mold_dims)
        self.mold_dimensions = mold_dims
        ###Reading the region information row by row in a specific order
        ###The data format is important here and pretty rigid, refer to the csv's in data folder
        for i in range(len(rows)):
            current_row = rows[i]
            ##Getting the region no and region id
            region_no = i+1
            region_id = current_row[0]
            current_region_features = synth_region_features(region_id)

            ##Region Number, order in which the number occurs
            current_region_features.region_no = region_no
            # print("Region No is: ", current_region_features.region_no)
            
            ##Region Dimension
            region_dims = current_row[1]
            region_dims = region_dims.split(",")
            region_dims = np.asfarray(region_dims)
            current_region_features.region_dims = copy.deepcopy(region_dims)

            #Region Area
            current_region_features.region_area = region_dims[0] * region_dims[1]

            ###Region Neighbors
            region_neighbors = current_row[2]
            region_neighbors = region_neighbors.split(",")
            region_neighbors = np.array(region_neighbors).astype(str)
            split_indices = np.where(region_neighbors == 'x')[0]
            region_neighbors = np.delete(region_neighbors, split_indices)
            split_indices = split_indices - np.arange(0, len(split_indices))
            edgewise_neighbors = np.split(region_neighbors, split_indices)
            # print('Region Neighbors: ', edgewise_neighbors)

            ###Region Lengths
            neighbor_lengths = current_row[3]
            neighbor_lengths = neighbor_lengths.split(",")
            neighbor_lengths = np.asfarray(neighbor_lengths)
            # print('Neighbor Lengths: ', neighbor_lengths)
           

            # ##Angles with region
            # neighbor_angles = current_row[4]
            # neighbor_angles = neighbor_angles.split(",")
            # neighbor_angles = np.array(neighbor_angles).astype(str)
            # print('Neighbor Angles: ', neighbor_angles)
            
            region_angle = current_row[4]
            region_angle = float(region_angle)
            region_angle = np.deg2rad(region_angle)
            current_region_features.region_angle = copy.deepcopy(region_angle)
            # print("Regions Angle: ", region_angle)
            

            ##No of edges for the given region
            current_region_features.no_of_edges = len(edgewise_neighbors)
            # print("No of edges: ", current_region_features.no_of_edges)
            
            start_index = 0

            for j in range(len(edgewise_neighbors)):
                current_edge_neighbors = edgewise_neighbors[j]
                current_edge_neighbors_dims = neighbor_lengths[start_index: start_index + len(current_edge_neighbors)]
                
                current_region_features.edgewise_region_dims[str(j)] = edgewise_neighbors[j]

                for k in range(len(current_edge_neighbors)):           
                    # print('Current Region: ', current_region_features.region_id)
                    # print(' Neighbor: ', current_edge_neighbors[k])
                    # print(' Region Angles: ', region_angles)
                    # print('start index: ', start_index)
                    # print('k: ', k)
                    # if(current_edge_neighbors[k] != 'O'):
                    #     current_region_features.angle_with_region[str(current_edge_neighbors[k])] = float(neighbor_angles[start_index+k])
                    
                    current_region_features.region_neighbor_dims[str(current_edge_neighbors[k])] = current_edge_neighbors_dims[k]
                    

                start_index += len(edgewise_neighbors[j]) 
            

            # print('Current Region Boundaries', current_region_features.edgewise_boundary_dims)
            # print('Neighbor Region Dims', current_region_features.region_neighbor_dims)
            # print('Angles with region: ', current_region_features.angle_with_region)

            ###Corner Coords
            corner_coords = current_row[5]
            corner_coords = corner_coords.split(",")
            corner_coords = np.array(corner_coords).astype(str)
            split_indices = np.where(corner_coords == 'x')[0]
            corner_coords = np.delete(corner_coords, split_indices)
            split_indices = split_indices - np.arange(0, len(split_indices))
            corner_coords = np.split(corner_coords, split_indices)
            corner_coords = np.asfarray(corner_coords)
            # print('Corner Coords: ', corner_coords)

            current_region_features.corner_coords = copy.deepcopy(corner_coords)

            ###Region centroids
            current_region_centroid = np.average(corner_coords, axis=0)
            # print("Centroid Coordinates: ", current_region_centroid)
            current_region_centroid[0] = current_region_centroid[0]/mold_dims[0]
            current_region_centroid[1] = current_region_centroid[1]/mold_dims[1]
            current_region_features.region_centroid = copy.deepcopy(current_region_centroid)
                     
            ###Corner Neighbors
            corner_neighbors = current_row[6]
            corner_neighbors = corner_neighbors.split(",")
            corner_neighbors = np.array(corner_neighbors).astype(str)
            current_region_features.corner_neighbors = corner_neighbors

           
            ##Getting the curvature
            curvature = float(current_row[7])
            # print('Curvature: ', curvature)
            current_region_features.curvature = copy.deepcopy(curvature)

            
            # print('Current Region Boundaries', current_region_features.edgewise_region_dims)
            # print('Neighbor Region Dims', current_region_features.region_neighbor_dims)
            
            

            ##Z Max
            current_region_features.z_max = float(current_row[8])
            # print('Z Max: ', current_region_features.z_max)

            # print("Current Region Features: ", current_region_features)

            self.region_information_dict[str(region_id)] = copy.deepcopy(current_region_features)
            


        # print('Region Information Dictionary: ', self.region_information_dict)
        
        return

    def parse_graph_tile(self, filename):

        self.max_vertices = 1
        self.no_of_regions = 1

        ###Reading the CSV information
        [fields, rows] = utils.read_csv(filename)
        # print('Fields: ', fields)
        # print('Rows: ', rows)
        

        ##No of Regions in the given graph
        self.no_of_regions = len(rows)
        print('No of regions: ', self.no_of_regions)

        ####Defining the User Preferred Path
        user_preffered_path = rows[0][11]
        user_preffered_path = user_preffered_path.split(",")
        user_preffered_path = np.array(user_preffered_path).astype(str)
        # print("User Preferred Path:", user_preffered_path)

        self.user_preferred_path_alpha = user_preffered_path
        self.user_preferred_path_num = utils.convert_alpha_path_to_num_path(user_preffered_path)

        ##Total Window Dimensions
        window_dims = rows[0][10]
        window_dims = window_dims.split(",")
        # print(window_dims)
        window_dims = np.asfarray(window_dims)
        self.window_dimensions = window_dims
        
        # print('Alpha Path: ', self.user_preferred_path_alpha)
        # print('Num Path: ', self.user_preferred_path_num)
        # print('Function test num path: ', utils.convert_num_path_to_alpha_path(self.user_preferred_path_num))
        
        ###Reading the region information row by row in a specific order
        ###The data format is important here and pretty rigid, refer to the csv's in data folder
        for i in range(len(rows)):
            current_row = rows[i]
            ##Getting the region no and region id
            tile_no = i+1
            tile_id = current_row[0]
            current_tile_features = tile(tile_id)
            
            ##Region Number, order in which the number occurs
            current_tile_features.tile_no = tile_no
            # print("Tile No is: ", current_tile_features.tile_no)
            
            ##Getting the region dimensions
            tile_dims = current_row[1]
            tile_dims = tile_dims.split(",")
            tile_dims = np.array(tile_dims).astype(float)
            # print('Tile Dimensions: ', tile_dims)
            
            current_tile_features.tile_dims = copy.deepcopy(tile_dims)

            #Tile Area Perimeter
            current_tile_features.tile_area = tile_dims[0] * tile_dims[1]


            ###Tile Neighbors
            tile_neighbors = current_row[2]
            tile_neighbors = tile_neighbors.split(",")
            tile_neighbors = np.array(tile_neighbors).astype(str)
            split_indices = np.where(tile_neighbors == 'x')[0]
            tile_neighbors = np.delete(tile_neighbors, split_indices)
            split_indices = split_indices - np.arange(0, len(split_indices))
            edgewise_neighbors = np.split(tile_neighbors, split_indices)
            # print('Tile Neighbors: ', edgewise_neighbors)
            
            ##No of edges for the given region
            current_tile_features.no_of_edges = len(edgewise_neighbors)
            # print("No of edges: ", current_region_features.no_of_edges)
            

            ###Neighbors Lengths
            neighbor_lengths = current_row[3]
            neighbor_lengths = neighbor_lengths.split(",")
            neighbor_lengths = np.asfarray(neighbor_lengths)
            # print('Neighbor Lengths: ', neighbor_lengths)
            
        
            
            start_index = 0

            for j in range(len(edgewise_neighbors)):
                current_edge_neighbors = edgewise_neighbors[j]
                current_edge_neighbors_dims = neighbor_lengths[start_index: start_index + len(current_edge_neighbors)]
                
                current_tile_features.edgewise_boundary_dims[str(j)] = edgewise_neighbors[j]

                for k in range(len(current_edge_neighbors)):           
                    # print('Current Region: ', current_region_features.region_id)
                    # print(' Neighbor: ', current_edge_neighbors[k])
                    # print(' Region Angles: ', region_angles)
                    # print('start index: ', start_index)
                    # print('k: ', k)
                    current_tile_features.tile_neighbor_dims[str(current_edge_neighbors[k])] = current_edge_neighbors_dims[k]
                    

                start_index += len(edgewise_neighbors[j]) 
            

            # print('Current Region Boundaries', current_region_features.edgewise_boundary_dims)
            # print('Neighbor Region Dims', current_region_features.region_neighbor_dims)
            # print('Angles with region: ', current_region_features.angle_with_region)

            ###Corner Coords
            corner_coords = current_row[4]
            corner_coords = corner_coords.split(",")
            corner_coords = np.array(corner_coords).astype(str)
            split_indices = np.where(corner_coords == 'x')[0]
            corner_coords = np.delete(corner_coords, split_indices)
            split_indices = split_indices - np.arange(0, len(split_indices))
            corner_coords = np.split(corner_coords, split_indices)
            corner_coords = np.asfarray(corner_coords)
            # print('Corner Coords: ', corner_coords)

            current_tile_features.corner_coords = copy.deepcopy(corner_coords)
            
        
            
            ###Corner Neighbors
            corner_neighbors = current_row[5]
            corner_neighbors = corner_neighbors.split(",")
            corner_neighbors = np.array(corner_neighbors).astype(str)
            current_tile_features.corner_neighbors = corner_neighbors
            


            ##Edge 1 list
            edge_1_list = current_row[6]
            edge_1_list = edge_1_list.split(",")
            edge_1_list = np.asarray(edge_1_list).astype(str)
            # print('Edge1 List: ', edge_1_list)
            if(len(edge_1_list) == 1 and edge_1_list[0] == 'x'):
                edge_1_list = []
            else:
                edge_1_list_indices = np.where(edge_1_list == 'x')[0]
                # print('List Indices: ', edge_1_list_indices)

            
                edge_1_list = np.delete(edge_1_list, edge_1_list_indices)
                edge_1_list_indices = edge_1_list_indices - np.arange(0, len(edge_1_list_indices))
                edge_1_list = np.split(edge_1_list, edge_1_list_indices)
                
            
            # print("Edge 1 List Final: ", edge_1_list)
            current_tile_features.edgewise_tile_neighbors[str(1)] = edge_1_list

            ##Edge 2 list
            edge_2_list = current_row[7]
            edge_2_list = edge_2_list.split(",")
            edge_2_list = np.asarray(edge_2_list).astype(str)
            # print('Edge2 List: ', edge_2_list)
            if(len(edge_2_list) == 1 and edge_2_list[0] == 'x'):
                edge_2_list = []
            else:
                edge_2_list_indices = np.where(edge_2_list == 'x')[0]
                # print('List Indices: ', edge_2_list_indices)

            
                edge_2_list = np.delete(edge_2_list, edge_2_list_indices)
                edge_2_list_indices = edge_2_list_indices - np.arange(0, len(edge_2_list_indices))
                edge_2_list = np.split(edge_2_list, edge_2_list_indices)
                
            
            # print("Edge 2 List Final: ", edge_2_list)
            current_tile_features.edgewise_tile_neighbors[str(2)] = edge_2_list

            ##Edge 3 list
            edge_3_list = current_row[8]
            edge_3_list = edge_3_list.split(",")
            edge_3_list = np.asarray(edge_3_list).astype(str)
            # print('Edge3 List: ', edge_3_list)
            if(len(edge_3_list) == 1 and edge_3_list[0] == 'x'):
                edge_3_list = []
            else:
                edge_3_list_indices = np.where(edge_3_list == 'x')[0]
                # print('List Indices: ', edge_3_list_indices)

            
                edge_3_list = np.delete(edge_3_list, edge_3_list_indices)
                edge_3_list_indices = edge_3_list_indices - np.arange(0, len(edge_3_list_indices))
                edge_3_list = np.split(edge_3_list, edge_3_list_indices)
                
            
            # print("Edge 3 List Final: ", edge_3_list)
            current_tile_features.edgewise_tile_neighbors[str(3)] = edge_3_list

            ##Edge 4 list
            edge_4_list = current_row[9]
            edge_4_list = edge_4_list.split(",")
            edge_4_list = np.asarray(edge_4_list).astype(str)
            # print('Edge4 List: ', edge_4_list)
            if(len(edge_4_list) == 1 and edge_4_list[0] == 'x'):
                edge_4_list = []
            else:
                edge_4_list_indices = np.where(edge_4_list == 'x')[0]
                # print('List Indices: ', edge_4_list_indices)

            
                edge_4_list = np.delete(edge_4_list, edge_4_list_indices)
                edge_4_list_indices = edge_4_list_indices - np.arange(0, len(edge_4_list_indices))
                edge_4_list = np.split(edge_4_list, edge_4_list_indices)
                
            
            # print("Edge 4 List Final: ", edge_4_list)
            current_tile_features.edgewise_tile_neighbors[str(4)] = edge_4_list


            self.region_information_dict[str(tile_id)] = copy.deepcopy(current_tile_features)
            


        # print('Region Information Dictionary: ', self.region_information_dict)
        
        return

    ###This method encodes a node/state based on which region/grid is visited
    ### 1 means a region is visited and 0 means the region is not visited
    def generate_state_space(self, n):
        
        if(n < 1):
            self.state_ids = np.append(self.state_ids, self.state_id_vector)
            
        else:
              
            self.state_id_vector[n-1] = int(0)
            self.generate_state_space(n- 1)
            self.state_id_vector[n-1] = int(1)
            self.generate_state_space(n - 1)

    ##Initialize Graph and add vertex and edges
    def initialize_graph(self):

        self.add_vertex(self.state_ids[0])

        for i in range(len(self.state_ids)):
            current_node = self.state_ids[i]
            current_node = current_node.astype(int)
            no_of_children = 0
            current_children = []
            zero_indices = np.where(current_node == 0)[0]
            if(len(zero_indices) == 0):
                self.add_vertex(current_node)
            for j in range(len(zero_indices)):
                current_child = np.zeros(self.no_of_regions, dtype=int)
                current_child[zero_indices[j]] = 1
                current_child = current_node + current_child
                self.add_vertex(current_child)
                self.add_edge(current_node,current_child)
                current_children = np.append(current_children, current_child)
                no_of_children+=1

        return

    ##This function adds a vertex to the graph
    ##Each vertex is meant to be state encoded as 0's and 1's
    def add_vertex(self, vertex_id):
        # print(vertex_id)
        if str(vertex_id) in self.vert_dict.keys():
            # print('Returned')
            return
        
        new_vertex = Node(str(vertex_id))
        
        self.vert_dict[str(vertex_id)] = new_vertex
        
        self.num_vertices += 1
        
        if(self.num_vertices == self.max_vertices):
            self.graph_initialized = True
        
        return

    def add_edge(self, source, destination):
        edge_params = self.evaluate_edge_cost(source, destination)
        
        #This is the vertex dictionary
        self.vert_dict[str(source)].add_child(destination,edge_params[0]) #Cost
        self.vert_dict[str(source)].define_child_features(destination,edge_params[1]) #Feature Vector

        return

    ##This function computes the edge cost between source and destination
    def evaluate_edge_cost(self, source,destination):
        
        if(self.graph_id == "tile"):
            feature_vector = feature_utilities.evaluate_tile_features(source,destination,self.region_information_dict, self.window_dimensions, self.no_of_features)
        elif(self.graph_id == "real"):
            feature_vector = feature_utilities.evaluate_features(source,destination,self.region_information_dict, self.no_of_features)
        elif(self.graph_id == "synthetic"):
            feature_vector = feature_utilities.evaluate_synthetic_features(source,destination,self.region_information_dict, self.mold_dimensions, self.no_of_features)

        self.no_of_features = len(feature_vector)
        if(self.no_of_features != len(self.features_weights)):
            print('Randomly initializing weights to match feature vector')
            self.features_weights = np.random.rand(self.no_of_features)
        
        cost = np.dot(self.features_weights, np.transpose(feature_vector))
        
        return [cost, feature_vector]

    
    #This recalculates the graph and reinitializes the edge weights given a new weight vector
    def reinitialize_weights(self, weight_vector):
        # print("Reinitializing Weights")
        # weight_vector = weight_vector/np.linalg.norm(weight_vector)
        self.features_weights = copy.deepcopy(weight_vector)

        for keys in self.vert_dict.keys():
            # print('Parent: ', keys)
            children = self.vert_dict[str(keys)].get_children_ids()

            for child in children:
                # print('Child', child)
                feature_vector = copy.deepcopy(self.vert_dict[str(keys)].get_child_features(child))
                new_edge_weight = np.dot(self.features_weights, feature_vector)
                self.vert_dict[str(keys)].add_child(child, new_edge_weight)
                
        return



if __name__ == '__main__':

    current_directory = os.path.dirname(__file__)
    
    main_directory = os.path.split(current_directory)[0]
    data_directory = main_directory + '/data/'
    

    # ###############################Tiling####################################
    # no_of_windows = 0
    # start_index = 1
    # # skip_data = [4,5,8,11,12,13]
    # skip_data = []
    # end_index = 10
    # for i in range (start_index,end_index+1,1):
    #     if(i in skip_data):
    #         continue
    #     no_of_windows += 1
    #     current_window_filename = copy.deepcopy(data_directory + "window_data_"+str(i))+".csv"
    #     print("Filname Passed: ", current_window_filename)

        
        
    #     current_window = copy.deepcopy(Graph(current_window_filename, "tile"))
    #     print('Done Initializing Window ', i)
    #     print("Weights used: ", current_window.features_weights)
    #     current_mold_dijstra = utils.shortest_path_dijstra(current_window.vert_dict, current_window.source_node)
    #     current_mold_shortest_path = utils.get_dijstra_result(current_window.source_node, current_window.goal_node, current_mold_dijstra[1])
    #     current_mold_shortest_path_alpha = utils.convert_num_path_to_alpha_path(current_mold_shortest_path[0])
    #     path_features = utils.getPathFeatures(current_window.vert_dict, current_mold_shortest_path[0])
    #     print('Data',i, ' shortest path: ', current_mold_shortest_path_alpha)
    #     print('Path Features: ', path_features)
    #     print(" ")
    #     del(current_window)

     ###############################Synthetic####################################
    no_of_molds = 0
    start_index = 1
    # skip_data = [4,5,8,11,12,13]
    # skip_data = [1,6,7,10]
    skip_data = []
    end_index = 10
    weights = [0.1682104,0.1,0.1,0.51383104,12.52011943,1.51917661,1.51917553,0.1,0.1,0.99206719]
    weights = [0.1,0.1,0.1,0.1,11.94215076,0.81442757,0.81442757,0.25408396,0.10259558,0.53996292]
    weights = [0.10873899,0.1,0.1,0.37013005,12.29598123,1.03473685,1.03473777,0.18743145,0.1,0.89889657]
    weights = [0.10033939,0.1,0.1, 0.24221834,12.24409938,0.97799328,0.97799328,0.11541914,0.1,0.93328917]
    weights = [0.1,0.1,0.1,0.1, 11.36042195,1.09174561,1.09174685,0.45352931,0.1,0.80500503]
    # weights = [0.1, 0.1,0.1,  0.1, 15.34361401, 1.16185861,  1.16185861,  0.37129504,  0.13124041,  0.51450431]
    # weights = np.ones(10)
    # weights = [0.1, 0.1,0.1, 0.1,12.10877947,0.95842931,0.95842931,0.48957086,0.1,0.86771969]
    # weights = [0.1, 0.12436435,0.1,0.29021553,10.57673174,1.35749638,1.35749475,1.63712692,0.10953099,0.1]
    
    #####################Effort Based Preference Results######################

    # effort_results_filename = data_directory + 'effort_preference_results.csv'

    # utils.plot_effort_pref_results(effort_results_filename, 'effort_results.png')
    # min_term_filename = data_directory +'shortest_path_sequence.csv'

    # utils.plot_min_term_results(min_term_filename, 'dummp.csv')

    avg_term_filename = data_directory +'mold_3_cost_data.csv'

    utils.scatter_plot_avg(avg_term_filename)



    # for i in range (start_index,end_index+1,1):
    #     if(i in skip_data):
    #         continue
    #     no_of_molds += 1
    #     current_data_filename = copy.deepcopy(data_directory + "synthetic_data_"+str(i))+".csv"
    #     print("Filname Passed: ", current_data_filename)

        
        
    #     current_mold = copy.deepcopy(Graph(current_data_filename, "synthetic"))
    #     print('Done Initializing Window ', i)
    #     current_mold.reinitialize_weights(weights)
    #     print("Weights used: ", current_mold.features_weights)
    #     current_mold_dijstra = utils.shortest_path_dijstra(current_mold.vert_dict, current_mold.source_node)
    #     current_mold_shortest_path = utils.get_dijstra_result(current_mold.source_node, current_mold.goal_node, current_mold_dijstra[1])
    #     current_mold_shortest_path_alpha = utils.convert_num_path_to_alpha_path(current_mold_shortest_path[0])
    #     path_features = utils.getPathFeatures(current_mold.vert_dict, current_mold_shortest_path[0])
    #     print('Data',i, ' shortest path: ', current_mold_shortest_path_alpha)
    #     print('Path Features: ', path_features)
    #     user_preferred_path_cost = utils.getPathCost(current_mold.vert_dict, current_mold.user_preferred_path_num)
    #     current_mold_path_cost = utils.getPathCost(current_mold.vert_dict, current_mold_shortest_path[0])
    #     print("User Preferred Path Cost: ", user_preferred_path_cost)
    #     print("Testing Shortest Path Cost: ", current_mold_path_cost)
    #     print(" ")
        
        
        # ################Power Analysis########################
        # current_mold_no_of_regions = len(current_mold_shortest_path_alpha)
        # max_number_of_paths = math.factorial(current_mold_no_of_regions)
        # loop_stop = 40000
        # if(current_mold_no_of_regions<8):
        #     loop_stop = max_number_of_paths

        # elif(current_mold_no_of_regions >= 9):
        #     loop_stop = 80000
        
        # current_mold_cost_vec =  user_preferred_path_cost
        # initial_traj = np.arange(1, current_mold_no_of_regions+1)
        # sampled_dict = {}
        # sample_count = 1
        
        # print("Beginning While with loop stop: ", loop_stop)
        # while(sample_count<loop_stop):
        #     current_sampled_trajectory = np.random.permutation(initial_traj)

        #     if(str(current_sampled_trajectory) in sampled_dict.keys()):
        #         continue
                
        #     current_sample_cost = utils.getPathCost(current_mold.vert_dict, current_sampled_trajectory)
        #     sampled_dict[str(current_sampled_trajectory)] = True
        #     sample_count += 1
        #     current_mold_cost_vec = np.append(current_mold_cost_vec, current_sample_cost) 
        #     # print(sample_count)
        
        # sampled_dict["Cost"] = current_mold_cost_vec
        # sampled_dict["Sample"] = np.arange(1,len(current_mold_cost_vec)+1)
        # sampled_dict["No of regions"] = current_mold_no_of_regions
        # sampled_dict["Sample Size"] = loop_stop
        # image_filename = "Mold_"+str(i)+"_Power.png"
        
 
        # utils.plot_scatter_power_spectrum(sampled_dict,"Cost",  "Sample", image_filename)
        
        
        # del(current_mold)

    # source = np.zeros(window1.no_of_regions, dtype=int)
    # destination = np.ones(window1.no_of_regions, dtype=int)
    
    # utils.find_k_shortest_path_brute(10, mold1)
    # print('Source: ', source)
    # print('Destination: ', destination)

    # # mold1.evaluate_edge_cost(source, destination)
    # print('Begginning K Shortest Paths')
    # mold_k_shortest_path = utils.k_Shortest_Paths(4,window1.vert_dict,source,destination, window1.user_preferred_path_num) 
    # start_time = timeit.default_timer()
    # # mold_k_shortest_path = utils.find_k_shortest_path_brute(10, mold1)
    # end_time = timeit.default_timer()

    # total_run_time = end_time - start_time

    # print('Total time: ', total_run_time)

    # for j in range(len(mold_k_shortest_path)):
    #     current_cost = utils.getPathCost(window1.vert_dict, mold_k_shortest_path[j])
    #     print('Cost of ', j, 'th shortest path ', mold_k_shortest_path[j], ' : ', current_cost)
    #     print('\n')