#Author: Mahalia Miller
#Date: February 10, 2014
#Edited by: Gitanjali Bhattacharjee
#LATEST UPDATE: 3 February 2020
#Edited on: 1 October 2019
#Latest edits: Returning multiple costs along with raw metrics of average_total_flow and average_travel_time.

#import some relevant Python packages
import pickle, random, pdb, time, networkx, pp, csv, numpy as np
from scipy.stats import norm
from math import log

#import some of my own custom packages
import util
import bd_test as bd
import ita_cost


def compute_flow(damaged_graph):
	'''compute max flow between a start and end'''
	# s=  'sf' #another option is '1000001'
	# t = 'oak' #other options are '1000002' and 'sfo'

	#print 'in compute flow'

	# GB ADDITION
	s = '1000027'
	t = '1000028'
	try:
		flow = networkx.max_flow(damaged_graph, s, t, capacity='capacity') #not supported by multigraph
	except networkx.exception.NetworkXError as e:
		print 'found an ERROR: ', e
		pdb.set_trace()
	return flow

def compute_shortest_paths(damaged_graph, demand):
	return -1

def compute_tt_vmt(damaged_graph, demand):

	it = ita_cost.ITA(damaged_graph,demand)
	newG, trips_made = it.assign() # GB MODIFICATION -- was previously just returning newG here

	travel_time = util.find_travel_time(damaged_graph) # GB QUESTION -- why do we put in damaged_graph rather than newG, since the latter actually has traffic assigned?
	vmt = util.find_vmt(damaged_graph)

	''' in the undamaged case, this should be around 172 million (http://www.mtc.ca.gov/maps_and_data/datamart/stats/vmt.htm) 
	over the course of a day, so divide by 0.053 (see demand note in main). BUT our trip table has only around 11 
	million trips (instead of the 22 million mentioned here: http://www.mtc.ca.gov/maps_and_data/datamart/stats/baydemo.htm 
	because we are looking at vehicle-driver only and not transit, walking, biking, being a passenger in a car, etc. 
	So, that's **8-9 million vehicle-miles divided by 2, which equals around 4 million vehicle-miles!**
	'''

	return travel_time, vmt, trips_made
	#return damaged_graph, travel_time, vmt, total_flow


def add_superdistrict_centroids(G):

	'''adds 34 dummy nodes for superdistricts'''
	sd_table = util.read_2dlist('input/superdistricts_clean.csv', ',', False)
	#for each superdistrict, create a dummy node. Make 2 directed edges from the dummy node to real nodes. Make 2 directed edges from real edges to dummy nodes.
	for row in sd_table:
		i = int(row[0])
		G.add_node(str(1000000 + i)) # create a superdistrict dummy node
		G.add_edge(str(1000000 + i), str(row[1]), capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
		G.add_edge(str(1000000 + i), str(row[2]), capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
		G.add_edge(str(row[3]), str(1000000 + i), capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
		G.add_edge(str(row[4]), str(1000000 + i), capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in

	#add a sf dummy node, an oakland dummy node, and a SFO dummy node for max flow
	G.add_node('sf')
	G.add_node('oak')
	G.add_node('sfo')
	G.add_edge('sf', '1000001', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('sf', '1000002', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('sf', '1000003', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('sf', '1000004', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('sf', '1000005', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('1000018', 'oak', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('1000019', 'oak', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('1000020', 'oak', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('6564', 'sfo', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('6563', 'sfo', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('6555', 'sfo', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('9591', 'sfo', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('6550', 'sfo', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in
	G.add_edge('6599', 'sfo', capacity_0 = 100000,  capacity = 100000, lanes =1 , bridges=[], distance_0=1, distance = 1, t_a=1, t_0=1, flow=0, dailyvolume=1) #capacity in vehicles over all lanes, travel time in seconds, length in miles, flow in

	return G

# -------------------------------- M. MILLER ORIGINAL DAMAGE_BRIDGES FUNCTION ------------------------------------------
# def damage_bridges(scenario, master_dict):
# 	'''This function damages bridges based on the ground shaking values (demand) and the structural capacity (capacity). It returns two lists (could be empty) with damaged bridges (same thing, just different bridge numbering'''
# 	from scipy.stats import norm
# 	damaged_bridges_new = []
# 	damaged_bridges_internal = []
#
# 	#first, highway bridges and overpasses
# 	beta = 0.6 #you may want to change this by removing this line and making it a dictionary lookup value 3 lines below
# 	for site in master_dict.keys(): #1-1889 in Matlab indices (start at 1)
# 		lnSa = scenario[master_dict[site]['new_id'] - 1]
# 		prob_at_least_ext = norm.cdf((1/float(beta)) * (lnSa - math.log(master_dict[site]['ext_lnSa'])), 0, 1) #want to do moderate damage state instead of extensive damage state as we did here, then just change the key name here (see master_dict description)
# 		U = random.uniform(0, 1)
# 		if U <= prob_at_least_ext:
# 			damaged_bridges_new.append(master_dict[site]['new_id']) #1-1743
# 			damaged_bridges_internal.append(site) #1-1889
# 	#num_damaged_bridges = sum([1 for i in damaged_bridges_new if i <= len(master_dict.keys())])
#
# 	# GB ADDDITION -- to use with master_dict = napa_dict, since napa_dict only has 40 bridges
# 	num_damaged_bridges = sum([1 for i in damaged_bridges_new if i <= 1743])
#
# 	return damaged_bridges_internal, damaged_bridges_new, num_damaged_bridges

# MODIFIED FUNCTION VERSION -- GB
def damage_bridges(scenario, master_dict, u):

	'''This function damages bridges based on the ground shaking values (demand) and the structural capacity (capacity). It returns two lists (could be empty) with damaged bridges (same thing, just different bridge numbering'''
	from scipy.stats import norm
	damaged_bridges_new = []
	damaged_bridges_internal = []

	#first, highway bridges and overpasses
	beta = 0.6 #you may want to change this by removing this line and making it a dictionary lookup value 3 lines below
	i = 0 # counter for bridge index
	for site in master_dict.keys(): #1-1889 in Matlab indices (start at 1)
		lnSa = scenario[master_dict[site]['new_id'] - 1]
		prob_at_least_ext = norm.cdf((1/float(beta)) * (lnSa - math.log(master_dict[site]['ext_lnSa'])), 0, 1) #want to do moderate damage state instead of extensive damage state as we did here, then just change the key name here (see master_dict description)
		if u[i] <= prob_at_least_ext:
			damaged_bridges_new.append(master_dict[site]['new_id']) #1-1743
			damaged_bridges_internal.append(site) #1-1889
		i += 1 # increment bridge index
	# GB ADDDITION -- to use with master_dict = napa_dict, since napa_dict only has 40 bridges
	num_damaged_bridges = sum([1 for i in damaged_bridges_new if i <= 1743])

	return damaged_bridges_internal, damaged_bridges_new, num_damaged_bridges

# # M. MILLER ORIGINAL FUNCTION
# def compute_damage(scenario, master_dict, index):
# 	'''goes from ground-motion intensity map to damage map '''
# 	#figure out component damage for each ground-motion intensity map
# 	damaged_bridges_internal, damaged_bridges_new, num_damaged_bridges = damage_bridges(scenario, master_dict) #e.g., [1, 89, 598] #num_bridges_out is highway bridges only
# 	return index, damaged_bridges_internal, damaged_bridges_new, num_damaged_bridges

# GB MODIFIED VERSION
def compute_damage(scenario, master_dict, index, U):
	'''goes from ground-motion intensity map to damage map '''
	#figure out component damage for each ground-motion intensity map

	damaged_bridges_internal, damaged_bridges_new, num_damaged_bridges = damage_bridges(scenario, master_dict, U) #e.g., [1, 89, 598] #num_bridges_out is highway bridges only
	return index, damaged_bridges_internal, damaged_bridges_new, num_damaged_bridges

def damage_highway_network(damaged_bridges_internal, G, master_dict, index):
	'''damaged bridges is a list of the original ids (1-1889, not the new ids 1-1743!!!!!!!) '''

	biggest_id_of_interest = max([int(k) for k in master_dict.keys()])
	road_bridges_out = sum([1 for i in damaged_bridges_internal if int(i) <= biggest_id_of_interest])
	try:
		if len(damaged_bridges_internal) > 0:
			b = damaged_bridges_internal[0].lower()
	except AttributeError:
		raise('Sorry. You must use the original ids, which are strings')

	list_of_u_v = []

	for site in damaged_bridges_internal:
		if int(site) <= 1889: #in original ids, not new ones since that is what is in the damaged bridges list
			affected_edges = master_dict[site]['a_b_pairs_direct'] + master_dict[site]['a_b_pairs_indirect']
			list_of_u_v += affected_edges
			for [u,v] in affected_edges:
				G[str(u)][str(v)]['t_0'] = float('inf')
				G[str(u)][str(v)]['t_a'] = float('inf')
				G[str(u)][str(v)]['capacity'] = 0 # so then in ita.assign(), the trips will be lost
				G[str(u)][str(v)]['distance'] = 20*G[str(u)][str(v)]['distance_0']

	return G, road_bridges_out

def measure_performance(damaged_graph, demand):

	travel_time, vmt, trips_made= compute_tt_vmt(damaged_graph, demand)

	return travel_time, vmt, trips_made

def compute_road_performance(G, damaged_bridges_internal, demand, no_damage_travel_time, no_damage_vmt, no_damage_trips_made, master_dict, index):
	'''computes network performance after damaging the network based on which bridges are damaged'''

	if G == None:

		G = get_graph()

	if len(damaged_bridges_internal) > 0:
		G, road_bridges_out = damage_highway_network(damaged_bridges_internal, G, master_dict, index)
		travel_time, vmt, trips_made = measure_performance(G, demand)
		G = util.clean_up_graph(G) #absolutely critical. otherwise, damage from scenario gets added to damage from previous scenarios!

	else: #no bridges are damaged, so no need to do all the calculations

		travel_time = no_damage_travel_time
		vmt = no_damage_vmt
		road_bridges_out = 0
		trips_made = no_damage_trips_made

	return index, road_bridges_out, travel_time, vmt, trips_made, damaged_bridges_internal

# def get_graph(): # M. MILLER ORIGINAL METHOD -- now deprecated
#
# 	import networkx
#
# 	'''loads full mtc highway graph with dummy links and then adds a few fake centroidal nodes for max flow and traffic assignment'''
# 	G = networkx.read_gpickle("input/graphMTC_CentroidsLength3int.gpickle")
# 	G = add_superdistrict_centroids(G)
# 	assert not G.is_multigraph() # Directed! only one edge between nodes
# 	G = networkx.freeze(G) #prevents edges or nodes to be added or deleted
# 	return G

def get_graph(): # GB UPDATED METHOD -- loads graph that has been corrected, pruned, and already had superdistrict centroids added to it

	import networkx

	G = networkx.read_gpickle("input/graphMTC_GB.gpickle") # corrected, pruned, and w/centroids added version of original graph in "input/graphMTC_CentroidsLength3int.gpickle"
	assert not G.is_multigraph() # Directed! only one edge between nodes
	G = networkx.freeze(G) #prevents the addition or deletion of edges or nodes
	return G

def save_results(bridge_array_internal, bridge_array_new, travel_index_times, numeps, seed):
	util.write_2dlist('output/' + time.strftime("%Y%m%d")+'_bridges_flow_path_tt_vmt_bridges_allBridges_roadonly_' + str(numeps) + 'eps_extensive_seed' + str(seed) +'.txt',travel_index_times)
	with open ('output/' + time.strftime("%Y%m%d")+'_' + str(numeps) + 'sets_damagedBridgesInternal_roadonly_seed' + str(seed) +'.pkl', 'wb') as f:
	  pickle.dump(bridge_array_internal, f)
	with open ('output/' + time.strftime("%Y%m%d")+'_' + str(numeps) + 'sets_damagedBridgesNewID_roadonly_seed' + str(seed) +'.pkl', 'wb') as f:
	  pickle.dump(bridge_array_new, f)

def save_results_0(bridge_array_internal, bridge_array_new, numeps, seed):
	with open ('output/' + time.strftime("%Y%m%d")+'_' + str(numeps) + 'sets_damagedBridgesInternal_roadonly_eed' + str(seed) +'temp.pkl', 'wb') as f:
	  pickle.dump(bridge_array_internal, f)
	with open ('output/' + time.strftime("%Y%m%d")+'_' + str(numeps) + 'sets_damagedBridgesNewID_roadonly_seed' + str(seed) +'temp.pkl', 'wb') as f:
	  pickle.dump(bridge_array_new, f)

# ----------------------------------- START OF GB ADDITIONS ------------------------------------------------------------

def new_to_oldIDs(master_dict):
	# takes in the master bridge dict
	# returns a dict with keys = bridge new IDs and values = bridge original IDs
	new_to_oldIDs = {}
	for k, v in master_dict.items():
		for v2 in v.keys():
			if v2 == 'new_id':
				n = master_dict[k][v2]
				new_to_oldIDs[str(n)] = k  # key = newID, value = originalID

	return new_to_oldIDs

def sample_gm_map(num_scenarios):
	# randomly choose which ground motion map to use

	gm_map_index = np.random.randint(0,num_scenarios)

	return gm_map_index

# def compute_average_performance(lnsas, map_weights, num_damage_maps, travel_times, vmts, trips_made, no_damage_travel_time, no_damage_vmt, no_damage_trips_made): #TODO -- OLD -- incorrect assumption about order of job and therefore of results
#
# 	# GB ADDITION -- computed weighted average (expectation) of travel time -- assumes travel times are stored in order
# 	average_travel_time = 0
# 	average_trips_made = 0
# 	average_vmt = 0
#
# 	for j in range(0, len(lnsas)):  # for every scenario considered
# 		w = map_weights[j]
# 		average_travel_time += w * np.average(np.asarray(travel_times[num_damage_maps * j:num_damage_maps * (
# 				j + 1)]))  # get the average travel time for that scenario
# 		average_trips_made += w * np.average(np.asarray(trips_made[num_damage_maps * j:num_damage_maps * (j + 1)]))
# 		average_vmt += w * np.average(np.asarray(vmts[num_damage_maps * j:num_damage_maps * (j + 1)]))
#
# 	# add the scenario of no earthquake
# 	average_travel_time += (1 - sum(map_weights)) * no_damage_travel_time
# 	average_trips_made += (1 - sum(map_weights)) * no_damage_trips_made
# 	average_vmt += (1 - sum(map_weights)) * no_damage_vmt
#
# 	return average_travel_time, average_vmt, average_trips_made

def compute_weighted_average_performance(lnsas, map_weights, num_damage_maps, travel_times, vmts, trips_made,
										 no_damage_travel_time, no_damage_vmt, no_damage_trips_made, direct_costs, alpha = 48, beta = 78*8):
	# Compute weighted average of performance metrics for a single sample of a fragility function vector.

	scenarios = len(lnsas) # number of scenarios under consideration

	# GB ADDITION -- computed weighted average (expectation) of travel time and other metrics of interest
	average_travel_time = 0
	average_trips_made = 0
	average_vmt = 0
	average_direct_costs = 0

	# convert input lists to arrays for multiple indexing
	travel_times = np.asarray(travel_times)
	trips_made = np.asarray(trips_made)
	vmts = np.asarray(vmts)
	direct_costs = np.asarray(direct_costs)

	for j in range(0, len(lnsas)):  # for every scenario considered
		w = map_weights[j]
		temp_times = np.asarray(travel_times[np.arange(start=j, stop=scenarios*num_damage_maps, step=scenarios)])
		temp_trips = np.asarray(trips_made[np.arange(start=j, stop=scenarios*num_damage_maps, step=scenarios)])
		temp_vmts = np.asarray(vmts[np.arange(start=j, stop=scenarios*num_damage_maps, step=scenarios)])
		temp_direct_costs = np.asarray(direct_costs[np.arange(start=j, stop=scenarios*num_damage_maps, step=scenarios)])
		assert temp_trips.shape[0] == num_damage_maps, 'Error -- wrong number of trips.'
		assert temp_times.shape[0] == num_damage_maps, 'Error -- wrong number of times.'
		assert temp_vmts.shape[0] == num_damage_maps, 'Error -- wrong number of vmts.'
		average_travel_time += w *np.average(temp_times)
		average_trips_made += w *np.average(temp_trips)
		average_vmt += w*np.average(temp_vmts)
		average_direct_costs += w*np.average(temp_direct_costs)

	# add the scenario of no earthquake
	average_travel_time += (1 - sum(map_weights)) * no_damage_travel_time
	average_trips_made += (1 - sum(map_weights)) * no_damage_trips_made
	average_vmt += (1 - sum(map_weights)) * no_damage_vmt

	average_delay_cost = alpha*max(0,((average_travel_time - no_damage_travel_time) / 3600))  # travel times are in seconds, so convert to units of monetary units/hour*hours --> monetary units per day; assume travel times increase with damage

	average_connectivity_cost = beta*max(0, (no_damage_trips_made - average_trips_made))  # units of monetary units/hour*lost trips/day*hours/(trips*days)--> monetary units per day; assume total flows decrease with damage

	assert average_delay_cost >= 0, 'ERROR in compute_indirect_costs(): delay cost is negative.'
	assert average_connectivity_cost >= 0, 'ERROR in compute_indirect_costs(): connectivity cost is negative.'

	average_indirect_cost = average_delay_cost + average_connectivity_cost

	return average_travel_time, average_vmt, average_trips_made, average_direct_costs, average_delay_cost, average_connectivity_cost, average_indirect_cost

def compute_direct_costs(damaged_bridges, partial_dict, unit_cost = 293, rep_cost_ratio = 0.25):

	# compute the direct cost for each damage map
	direct_costs = np.zeros((len(damaged_bridges),))
	for i in range(0, len(damaged_bridges)): # for every damage map (S x D)
		temp_cost = 0
		temp_n = len(damaged_bridges[i])
		if temp_n > 0:
			for j in range(0, temp_n):
				temp_cost += partial_dict[str(damaged_bridges[i][j])]['area'] * unit_cost * rep_cost_ratio
		direct_costs[i] = temp_cost

	return direct_costs

def compute_indirect_costs(no_damage_travel_time, no_damage_trips_made, travel_times, trips_made, alpha = 48, beta = 78*8):

	delay_costs = [alpha*max(0,((travel_times[i] - no_damage_travel_time) / 3600)) for i in range(0, len(travel_times))] # units of current dollars
	connectivity_costs = [beta*max(0, (no_damage_trips_made - trips_made[i])) for i in range(0, len(trips_made))] # units of current dollars
	indirect_costs = [delay_costs[i] + connectivity_costs[i] for i in range(0,len(delay_costs))] # units of current dollars

	delay_costs = np.asarray(delay_costs)
	connectivity_costs = np.asarray(connectivity_costs)
	indirect_costs = np.asarray(indirect_costs)

	return delay_costs, connectivity_costs, indirect_costs  # units are current dollars

# def compute_expected_direct_costs(lnsas, map_weights, num_damage_maps, direct_costs): # OLD -- replaced by compute_weighted_average_performance()
#
# 	exp_direct_cost = 0
# 	for j in range(0, len(lnsas)):  # for every scenario considered
# 		w = map_weights[j]
# 		exp_direct_cost += w * np.average(np.asarray(direct_costs[num_damage_maps * j:num_damage_maps * (j + 1)]))
#
# 	return exp_direct_cost
#
# def compute_expected_indirect_cost(no_damage_travel_time, no_damage_trips_made, average_travel_time, average_trips_made, alpha = 48, beta = 78*8): # OLD -- replaced by compute_weighted_average_performance()
#
# 	delay_cost = alpha*max(0,((average_travel_time - no_damage_travel_time) / 3600))  # travel times are in seconds, so convert to units of monetary units/hour*hours --> monetary units per day; assume travel times increase with damage
#
# 	connectivity_cost = beta*max(0, (no_damage_trips_made - average_trips_made))  # units of monetary units/hour*lost trips/day*hours/(trips*days)--> monetary units per day; assume total flows decrease with damage
#
#
# 	assert delay_cost >= 0, 'ERROR in compute_indirect_costs(): delay cost is negative.'
# 	assert connectivity_cost >= 0, 'ERROR in compute_indirect_costs(): connectivity cost is negative.'
#
# 	indirect_cost = delay_cost + connectivity_cost
#
#
# 	return delay_cost, connectivity_cost, indirect_cost

def compute_retrofit_cost(bridge_ids, partial_dict, x, unit_cost = 293, rep_cost_ratio = 0.25, ret_cost_ratio = 0.30): # estimate retrofit cost for each bridge as 30% of the repair cost

	retrofit_cost = 0

	i = 0
	for b in bridge_ids: # for every bridge
		if x[i] > partial_dict[b]['ext_lnSa']: # then we know bridge has been retrofitted, incurring a retrofit cost -- we can use x[i] since we generated samples in same order as bridges appear in bridge_ids
			temp_cost = (partial_dict[b]['area'] * unit_cost * rep_cost_ratio)*ret_cost_ratio
			retrofit_cost += temp_cost
		i += 1

	return retrofit_cost


# ----------------------------------- END OF GB ADDITIONS --------------------------------------------------------------

def main_dam_only(sample_index, map_indices, map_weights, bridge_ids, partial_dict, x, U, demand, damage_tracker,
		 bridge_indices, no_damage_travel_time, no_damage_vmt,
		 no_damage_trips_made, num_gm_maps = 10, num_damage_maps = 3):
	# '''this is the main file that runs from ground-motion intensity map to network performance measure. You will  need to adjust various things below, such as the ground motion files, performance measure info and more. you should not need to change, however, the functions that they call'''

	seed_num = 0  # USER ADJUSTS THIS! other value examples: 1,2, 11, 14, ...
	random.seed(seed_num)  # set random number generator seed so we can repeat this process

	## GB: this gets hazard-consistent maps that we created from Miller's subsetting procedure
	sa_matrix_full = util.read_2dlist('input/sample_ground_motion_intensity_maps_road_only_filtered.txt',
									  delimiter='\t')
	sa_matrix = [sa_matrix_full[i] for i in
				 map_indices]  # GB: get the ground_motions for just the scenarios we are interested in

	lnsas = []
	magnitudes = []
	for row in sa_matrix:
		lnsas.append([log(float(sa)) for sa in row[4:]])
		magnitudes.append(float(row[2]))

	################## component (bridge) damage map data #######################
	sets = num_damage_maps  # number of bridge damage maps per ground-motion intensity map. USER ADJUSTS THIS! other value examples: 3,9,18
	targets = range(0, len(
		lnsas) * sets)  # define the damage map IDs you want to consider. Note: this currently does not require modification. Just change the number of sets above.

	i = 0  # bridge index counter
	for b in bridge_ids:  # GOOD -- set order the whole time -- be sure this is the same order used to generate the F samples
		partial_dict[b]['ext_lnSa'] = x[i]
		i += 1

	#################################################################
	################## actually run damage map creation #######################

	start = time.time()
	ppservers = ()  # starting a super cool parallelization
	# Creates jobserver with automatically detected number of workers
	job_server = pp.Server(ncpus=4, ppservers=ppservers)
	print "Starting pp with", job_server.get_ncpus(), "workers"
	# set up jobs

	jobs = []
	for i in range(0, len(lnsas)):
		j = 0
		while j < num_damage_maps:
			jobs.append(job_server.submit(compute_damage, (lnsas[i], partial_dict, targets[i], i, U[i*num_damage_maps + j][:]),
							  modules=('random', 'math',), depfuncs=(damage_bridges,)))
			j += 1

	# get the results that have already run
	bridge_array_new = []
	bridge_array_internal = []
	indices_array = []  # GB: stores index of damage map being considered (or GM intensity map? unclear)
	bridge_array_hwy_num = []  # GB:
	for job in jobs:
		(index, damaged_bridges_internal, damaged_bridges_new, num_damaged_bridges_road) = job()
		bridge_array_internal.append(damaged_bridges_internal)
		bridge_array_new.append(damaged_bridges_new)
		indices_array.append(index)
		bridge_array_hwy_num.append(num_damaged_bridges_road)


	# # update damage counter
	# for i in range(0, len(bridge_array_internal)):  # for each entry in the list, i.e. each damage map (n_scenarios * n_dam_maps_per_scenario)
	# 	# if len(targets) == 19920 and i == 600:
	# 	# 	print 'damaged bridges at 600 = ', bridge_array_internal[i]
	# 	# if len(targets) == 200 and i == 0:
	# 	# 	print 'damaged bridges at 0 = ', bridge_array_internal[i]
	# 	if len(bridge_array_internal[i]) > 0:  # if bridges were damaged on this map
	# 		for j in range(0, len(bridge_array_internal[i])):  # for each bridge that was damaged
	# 			damage_tracker[i][bridge_indices[str(bridge_array_internal[i][j])]][sample_index] += 1  # increment the
	# # damage counter; sample_index works because we run main() for a single sample of fragility function parameters
	for i in range(0, len(lnsas)):
		j = 0
		while j < num_damage_maps:
			if len(bridge_array_internal[i* num_damage_maps + j]) > 0:  # if bridges were damaged on this map
				for k in range(0, len(bridge_array_internal[i* num_damage_maps + j])):  # for each bridge that was damaged
					damage_tracker[i* num_damage_maps + j][bridge_indices[str(bridge_array_internal[i* num_damage_maps + j][k])]][sample_index] += 1
			j += 1


	job_server.destroy()

	print 'Creating damage maps took ', time.time() - start, ' seconds.'

	return damage_tracker

def main(sample_index, map_indices, map_weights, bridge_ids, partial_dict, x, U, demand, damage_tracker,
		 bridge_indices, no_damage_travel_time, no_damage_vmt,
		 no_damage_trips_made, num_gm_maps = 10, num_damage_maps = 3, retrofit_cost = False):
	# '''this is the main file that runs from ground-motion intensity map to network performance measure. You will  need to adjust various things below, such as the ground motion files, performance measure info and more. you should not need to change, however, the functions that they call'''

	seed_num = 0  # USER ADJUSTS THIS! other value examples: 1,2, 11, 14, ...
	random.seed(seed_num)  # set random number generator seed so we can repeat this process

	## GB: this gets hazard-consistent maps that we created from Miller's subsetting procedure
	sa_matrix_full = util.read_2dlist('input/sample_ground_motion_intensity_maps_road_only_filtered.txt',
									  delimiter='\t')
	sa_matrix = [sa_matrix_full[i] for i in
				 map_indices]  # GB: get the ground_motions for just the scenarios we are interested in


	lnsas = []
	magnitudes = []
	for row in sa_matrix:
		lnsas.append([log(float(sa)) for sa in row[4:]])
		magnitudes.append(float(row[2]))

	# compute bridge retrofit cost prior to altering partial_dict
	ret_cost = compute_retrofit_cost(bridge_ids, partial_dict, x)

	################## component (bridge) damage map data #######################
	sets = num_damage_maps  # number of bridge damage maps per ground-motion intensity map. USER ADJUSTS THIS! other value examples: 3,9,18
	targets = range(0, len(
		lnsas) * sets)  # define the damage map IDs you want to consider. Note: this currently does not require modification. Just change the number of sets above.

	i = 0  # bridge index counter
	for b in bridge_ids:  # GOOD -- set order the whole time -- be sure this is the same order used to generate the F samples
		partial_dict[b]['ext_lnSa'] = x[i]
		i += 1

	#################################################################
	################## actually run damage map creation #######################

	start = time.time()
	ppservers = ()  # starting a super cool parallelization
	# Creates jobserver with automatically detected number of workers
	job_server = pp.Server(ncpus=4, ppservers=ppservers)
	print "Starting pp with", job_server.get_ncpus(), "workers"
	# set up jobs

	jobs = []
	for i in targets:
		jobs.append(
			job_server.submit(compute_damage, (lnsas[i % len(lnsas)], partial_dict, targets[i], U[i][:]),
							  modules=('random', 'math',), depfuncs=(damage_bridges,)))

	# get the results that have already run
	bridge_array_new = []
	bridge_array_internal = []
	indices_array = []  # GB: stores index of damage map being considered (or GM intensity map? unclear)
	bridge_array_hwy_num = []  # GB:
	for job in jobs:
		(index, damaged_bridges_internal, damaged_bridges_new, num_damaged_bridges_road) = job()
		bridge_array_internal.append(damaged_bridges_internal)
		bridge_array_new.append(damaged_bridges_new)
		indices_array.append(index)
		bridge_array_hwy_num.append(num_damaged_bridges_road)

	# # update damage counter
	# for i in range(0, len(bridge_array_internal)):  # for each entry in the list, i.e. each damage map (n_scenarios * n_dam_maps_per_scenario)
	# 	if len(bridge_array_internal[i]) > 0:  # if bridges were damaged on this map
	# 		for j in range(0, len(bridge_array_internal[i])):  # for each bridge that was damaged
	# 			damage_tracker[i][bridge_indices[str(bridge_array_internal[i][j])]][sample_index] += 1  # increment the
	# # damage counter; sample_index works because we run main() for a single sample of fragility function parameters
	#
	for i in range(0, len(lnsas)):
		j = 0
		while j < num_damage_maps:
			if len(bridge_array_internal[i* num_damage_maps + j]) > 0:  # if bridges were damaged on this map
				for k in range(0, len(bridge_array_internal[i* num_damage_maps + j])):  # for each bridge that was damaged
					damage_tracker[i* num_damage_maps + j][bridge_indices[str(bridge_array_internal[i* num_damage_maps + j][k])]][sample_index] += 1
			j += 1

	job_server.destroy()

	print 'Creating damage maps took ', time.time() - start, ' seconds.'

	# #################################################################
	# ################## actually run performance measure realization creation #######################
	# START OF GB MODIFICATION -- batch processing rather than submitting all the jobs and waiting for them to run
	start = time.time()
	ppservers = ()
	# Creates jobserver with automatically detected number of workers
	job_server = pp.Server(ncpus=4, ppservers=ppservers)
	#print "Starting pp with", job_server.get_ncpus(), "workers"
	# set up jobs
	jobs = []

	# get the results that have already run and save them
	travel_times = []
	trips_made = []
	vmts = []
	damaged_bridges = []
	# made_flows = []
	for i in targets:
		jobs.append(job_server.submit(compute_road_performance, (None, bridge_array_internal[i], demand, no_damage_travel_time, no_damage_vmt, no_damage_trips_made, partial_dict, targets[i],),
									  modules=('networkx', 'time', 'pickle', 'pdb', 'util', 'random', 'math', 'ita_cost',),
									  depfuncs=(
										  get_graph, damage_bridges,
										  damage_highway_network,
										  measure_performance,
										  compute_tt_vmt,)))  # functions, modules


	# print 'STARTING TO PULL RESULTS'
	i = 0
	for job in jobs:
		# (index, road_bridges_out, flow, shortest_paths, travel_time, vmt) = job()
		(index, road_bridges_out, travel_time, vmt, trips, damaged_bridges_internal) = job()

		assert indices_array[i] == index, 'the damage maps should correspond to the performance measure realizations'
		assert bridge_array_hwy_num[i] == road_bridges_out, 'we should also have the same number of hwy bridges out'
		travel_times.append(travel_time)
		trips_made.append(trips)
		vmts.append(vmt)
		damaged_bridges.append(damaged_bridges_internal)
		# print 'job i = ', i, travel_time, lost_flow, road_bridges_out, damaged_bridges_internal
		#print 'done with job ', i
		#print i, travel_times[i], total_flows[i], vmts[i], damaged_bridges[i]
		i += 1

	job_server.destroy()

	traffic_time = time.time() - start
	print 'Assigning traffic took ', traffic_time, ' seconds.'

	# #################################################################
	# ################## compute the costs associated with the road network's performance(s) #######################

	# OLD -- replaced by compute_weighted_average_performance()
	# average_travel_time, average_vmt, average_trips_made = compute_average_performance(lnsas, map_weights,
	# 																				   num_damage_maps, travel_times,
	# 																				   vmts, trips_made,
	# 																				   no_damage_travel_time,
	# 																				   no_damage_vmt,
	# 																				   no_damage_trips_made) # should be scalars
	# expected_repair_cost = compute_expected_direct_costs(lnsas, map_weights, num_damage_maps, repair_costs) # should be a scalar

	# expected_delay_cost, expected_conn_cost, expected_indirect_cost = compute_expected_indirect_cost(no_damage_travel_time, no_damage_trips_made, average_travel_time, average_trips_made) # should be scalars

	# NOTE -- using average and expected interchangeably -- these are all weighted averages
	delay_costs, conn_costs, indirect_costs = compute_indirect_costs(no_damage_travel_time, no_damage_trips_made, travel_times, trips_made) # should have dimensions of (S x D,) -- these are raw data
	repair_costs = compute_direct_costs(damaged_bridges, partial_dict) # raw data
	average_travel_time, average_vmt, average_trips_made, expected_repair_cost, expected_delay_cost, \
	expected_conn_cost, expected_indirect_cost = compute_weighted_average_performance(lnsas, map_weights,
																					   num_damage_maps, travel_times,
																					   vmts, trips_made,
																					   no_damage_travel_time,
																					   no_damage_vmt,
																					   no_damage_trips_made, direct_costs=repair_costs)

	expected_total_cost = expected_repair_cost + expected_delay_cost + expected_conn_cost

	if retrofit_cost:
		expected_total_cost += ret_cost

	return travel_times, trips_made, vmts, average_travel_time, average_trips_made, average_vmt, damage_tracker, \
		   delay_costs, conn_costs, indirect_costs, repair_costs, \
		   expected_delay_cost, expected_conn_cost, expected_indirect_cost, expected_repair_cost, expected_total_cost, retrofit_cost # damaged_bridges # note that costs are also expectations! just not named as such
