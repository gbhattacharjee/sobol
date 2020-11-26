# Author: Gitanjali Bhattacharjee
# LATEST UPDATE: 30 January 2020 -- update to work with fixed traffic model and new cost model.
# LATEST UPDATE: copied from compute_bridge_sobol_multi_cost_alt_test.py -- major update is that now, samples are processed
# in parallel using pp package and damage maps and costs are computed in series -- MUCH FASTER THIS WAY!
# LATEST UPDATE: 24 August 2019 -- changed how we generate F and pass it to mahmodel (now pass bridge indices list as well,
# to preserve order and make sure nothing gets screwed up)
# LATEST UPDATE: 3 August 2019 -- changed how we generate U from producing an array of size (B,S) to one of size (B,SxD);
# changed reference to mahmodel_road_only_napa_3 (from v2)
# LATEST UPDATE: 30 July 2019 -- added 'first_order' as an optional parameter to compute first-order Sobol indices rather
# than only total-order Sobol indices -- changed interleave() method
# LATEST UPDATE: 7 July 2019 -- added 'map_indices' as an input to run_traffic_model_set to allow for custom subset of maps
# Purpose: This file is used in conjunction with mahmodel_road_only_napa.py for my ME 470 project. It contains code to
# explore inputs and conduct analysis.

from __future__ import division
import mahmodel_road_only as mahmodel
import util, pp, pickle, copy, numpy, time
import networkx as nx

from math import log
from matplotlib import container # just to get errorbar legend without vertical bars
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import bd_test as bd

def get_bridge_ids(cs_number):
	if cs_number == 1:
		filename = 'cs1_bridge_ids.pkl'
	elif cs_number == 2:
		filename = 'cs2_bridge_ids.pkl'
	elif cs_number == 3:
		filename = 'cs3_bridge_ids.pkl'
	elif cs_number == 4:
		filename = 'cs4_bridge_ids.pkl'
	elif cs_number == 5:
		filename = 'cs5_bridge_ids.pkl'
	elif cs_number == 6:
		filename = 'cs6_bridge_ids.pkl'
	elif cs_number == 7:
		filename = 'cs7_bridge_ids.pkl'
	elif cs_number == 8:
		filename = 'cs8_bridge_ids.pkl'
	elif cs_number == 9:
		filename = 'cs9_bridge_ids.pkl'
	else:
		filename = None
		print 'ERROR -- choose cs_number = 1, 2, or 3.'

	filepath = 'input/' + filename

	with open(filepath, 'rb') as f:
		bridge_ids = pickle.load(f)

	# print 'bridges = ', bridge_ids

	return bridge_ids

def get_master_dict():
	#with open('input/20140114_master_bridge_dict.pkl', 'rb') as f:
	with open('input/master_bridge_dict_GB_omegas.pkl', 'rb') as f: # this version includes area for computation of repair cost
		master_dict = pickle.load(f)  # has 1743 keys. One per highway bridge. (NOT BART)
		'''
		dict where the keyranges from 1 to 1889 and then the value is another dictionary with the following keys:
		loren_row_number: the row number in the original table that has info on all CA bridges (where the header line is row 0)
		original_id: the original id (1-1889)
		new_id: the new id that excludes filtered out bridges (1-1743). Bridges are filtered out if a.) no seismic capacity data AND non-transbay bridge or b.) not located by Jessica (no edge list). this id is the new value that is the column number for the lnsa simulations.
		jessica_id: the id number jessica used. it's also the number in arcgis.
		a_b_pairs_direct: list of (a,b) tuples that would be directly impacted by bridge damage (bridge is carrying these roads)
		a_b_pairs_indirect: ditto but roads under the indirectly impacted bridges
		edge_ids_direct: edge object IDS for edges that would be directly impacted by bridge damage
		edge_ids_indirect: ditto but roads under the indirectly impacted bridges
		mod_lnSa: median Sa for the moderate damage state. the dispersion (beta) for the lognormal distribution is 0.6. (See hazus/mceer method)
		ext_lnSa: median Sa for the extensive damage state. the dispersion (beta) for the lognormal distribution is 0.6. (See hazus/mceer method)
		com_lnSa: median Sa for the complete damage state. the dispersion (beta) for the lognormal distribution is 0.6. (See hazus/mceer method)
		'''
	return master_dict

def get_bridge_dict(cs_number):

	if cs_number == 1:
		filename = 'cs1_dict.pkl'
	elif cs_number == 2:
		filename = 'cs2_dict.pkl'
	elif cs_number == 3:
		filename = 'cs3_dict.pkl'
	elif cs_number == 4:
		filename = 'cs4_dict.pkl'
	elif cs_number == 5:
		filename = 'cs5_dict.pkl'
	elif cs_number == 6:
		filename = 'cs6_dict.pkl'
	elif cs_number == 7:
		filename = 'cs7_dict.pkl'
	elif cs_number == 8:
		filename = 'cs8_dict.pkl'
	elif cs_number == 9:
		filename = 'cs9_dict.pkl'
	else:
		filename = None
		print 'ERROR -- choose cs_number = 1, 2, or 3.'

	filepath = 'input/' + filename

	with open(filepath, 'rb') as f:
		bridge_dict = pickle.load(f)

	return bridge_dict

def create_partial_dict(bridges):

	n_bridges = len(bridges)
	master_dict = get_master_dict()
	partial_dict = {bridges[i]:master_dict[bridges[i]] for i in range(0,n_bridges)}

	return partial_dict

def get_partial_napa_bridge_dict(napa_dict, n_bridges):

	bridges = napa_dict.keys()
	partial_napa_dict = {bridges[i]:napa_dict[bridges[i]] for i in range(0,n_bridges)}

	return partial_napa_dict

def generate_uniform_numbers(n_bridges, n_damage_maps, filename, store = True):
	# takes as input:
	# n_bridges, the number of bridges of interest
	# n_damage_maps, (S x D) the number of scenarios (ground motion maps) times the number of damage maps per scenario to be considered
	# returns U, an array of size (n_damage_maps, n_bridges) in which each element is a sample from a standard uniform distribution

	U = numpy.random.uniform(size=(n_damage_maps, n_bridges))

	if store:
		with open('sobol_input/' + filename, 'wb') as f:
			pickle.dump(U, f)

	return U

def compute_sample_mean(Y_set):
	# takes as input a vector in which each element is the Y associated with a sample of X
	n_samples = Y_set.shape[0]
	mu_hat = sum(Y_set)/n_samples

	return mu_hat

def compute_sample_variance(Y_set):
	# takes as input a vector in which each element is the Y associated with a sample of X
	n_samples = Y_set.shape[0]
	mu_hat = compute_sample_mean(Y_set)
	var_hat = sum((Y_set -  mu_hat)**2)/(n_samples-1) # this is sigma-hat-squared

	return var_hat

def compute_CI(S_set, confidence, t_dist=False, return_mean = False):

	# takes as input S_set, a vector of estimated total-order Sobol' indices, and a desired confidence level between 0 and 1
	# optional parameter t_dist indicates whether user wants to use Student's t-distribution instead of the
	# standard normal distribution -- using the t-dist is for cases in which the true variance is not known
	# returns the size of the one-sided confidence interval, assuming that the mean of the Sobol' indices are normally distributed

	m = S_set.shape[0] # number of samples of Sobol' index
	alpha = 1 - confidence

	assert 0 <= alpha <= 1, 'error in confidence level'

	# get the sample mean, mu_hat, of the Sobol' indices
	mu_hat = compute_sample_mean(S_set)
	var_hat = compute_sample_variance(S_set)

	if t_dist:
		CI = stats.t.ppf(alpha/2,m-1)*numpy.sqrt(var_hat/m) # size of one-sided confidence interval
	else:
		CI = stats.norm.ppf(alpha/2)*numpy.sqrt(var_hat/m) # size of one-sided confidence interval

	if return_mean:
		return mu_hat, CI
	else:
		return CI

# --------------------------------------------- TRAFFIC EQUATIONS METHODS ----------------------------------------------

def get_sa_matrix(map_indices):

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

	return lnsas

# --------------------------------------------- DAMAGE MAPS METHODS ----------------------------------------------------

def iterate_original_damage_rates(scenarios, num_damage_maps = 10):
	import os

	# partial_dict_name = 'sf_testbed_2_dict'
	partial_dict_name = 'sft2_dict'
	# load bridge dict, fragility function parameter samples, and uniform random number samples
	partial_dict_path = 'input/' + partial_dict_name + '.pkl'
	with open(partial_dict_path, 'rb') as f:
		partial_dict = pickle.load(f)

	print 'bridges = ', partial_dict.keys()

	sf_testbed = [951, 1081, 935, 895, 947, 993, 976, 898, 925, 926, 966, 917, 1065, 953, 972, 990, 899, 919, 904,
				  940]  # see bridge_metadata_NBI_sf_tetsbed/sf_testbed_new_3 -- otherwise referred to as sf_testbed_2
	bridge_ids = [str(b) for b in sf_testbed]
	n_bridges = len(bridge_ids)
	bridge_indices = {bridge_ids[i]: i for i in
					  range(0, len(bridge_ids))}  # each bridge has an index in the damage_tracker array

	if scenarios == 20:
		map_indices_input = 'sobol_input/sf_testbed_2_map_indices.pkl'  # S = 20, original
		map_weights_input = 'sobol_input/sf_testbed_2_map_weights.pkl'  # S = 20
		output_folder = 'debugging_sobol/original_s20'
	elif scenarios == 40:
		map_indices_input = 'sobol_input/sf_testbed_2_testing_map_indices.pkl'  # S = 40, original
		map_weights_input = 'sobol_input/sf_testbed_2_testing_map_weights.pkl'  # S = 40
		output_folder = 'debugging_sobol/original_s40'

	with open(map_indices_input, 'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input, 'rb') as f:
		map_weights = pickle.load(f)

	print map_indices
	print map_weights

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

	print len(map_indices), map_indices
	print len(map_weights), map_weights

	assert len(
		map_indices) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map indices list.'
	assert len(
		map_weights) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map weights list.'

	f_unretrofitted = numpy.zeros((1, n_bridges))
	f_retrofitted = numpy.zeros((1, n_bridges,))
	i = 0
	for b in bridge_ids:
		f_unretrofitted[0, i] = partial_dict[b]['ext_lnSa']
		f_retrofitted[0, i] = 2 * partial_dict[b]['ext_lnSa']
		i += 1

	print 'f_unretrofitted', f_unretrofitted
	print 'f_retrofitted', f_retrofitted

	# Set up traffic model and run it.
	G = mahmodel.get_graph()

	assert G.is_multigraph() == False, 'You want a directed graph without multiple edges between nodes'

	demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
							 'input/superdistricts_centroids_dummies.csv')

	no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = precompute_network_performance(G, demand)

	print 'no_damage_travel_time = ', no_damage_travel_time
	print 'no_damage_vmt = ', no_damage_vmt
	print 'no_damage_trips_made = ', no_damage_trips_made

	sc = False
	while sc is False:

		U = generate_uniform_numbers(n_bridges, scenarios * num_damage_maps, 'U_temp.pkl', store=False)

		assert U.shape[
				   0] == scenarios * num_damage_maps, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
		assert U.shape[
				   1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'

		damage_tracker_unret = numpy.zeros((scenarios * num_damage_maps, n_bridges, 1))  # array of size (SxD, B, 1)
		damage_tracker_ret = numpy.zeros((scenarios * num_damage_maps, n_bridges, 1))  # array of size (SxD, B, 1)

		damage_tracker_unret_output = output_folder + 'damage_tracker_unret.pkl'
		damage_tracker_ret_output = output_folder + 'damage_tracker_ret.pkl'

		damage_tracker_unret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights,
															  f_unretrofitted, U,
															  demand, damage_tracker_unret, bridge_indices,
															  no_damage_travel_time,
															  no_damage_vmt, no_damage_trips_made,
															  num_gm_maps=scenarios,
															  num_damage_maps=num_damage_maps)

		damage_tracker_ret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights,
															f_retrofitted, U,
															demand, damage_tracker_ret, bridge_indices,
															no_damage_travel_time,
															no_damage_vmt, no_damage_trips_made, num_gm_maps=scenarios,
															num_damage_maps=num_damage_maps)

		with open(damage_tracker_unret_output, 'wb') as f:
			pickle.dump(damage_tracker_unret, f)
		with open(damage_tracker_ret_output, 'wb') as f:
			pickle.dump(damage_tracker_ret, f)

		print 'damage tracker shapes: ', damage_tracker_unret.shape, damage_tracker_ret.shape

		cond_damage_rates_unret = numpy.zeros((scenarios, n_bridges))
		cond_damage_rates_ret = numpy.zeros((scenarios, n_bridges))

		for s in range(0, scenarios):
			temp_unret = numpy.average(damage_tracker_unret[s * num_damage_maps:(s + 1) * num_damage_maps, :], axis=0)
			temp_ret = numpy.average(damage_tracker_ret[s * num_damage_maps:(s + 1) * num_damage_maps, :], axis=0)
			cond_damage_rates_unret[s, :] = numpy.reshape(temp_unret, (n_bridges))
			cond_damage_rates_ret[s, :] = numpy.reshape(temp_ret, (n_bridges))

		damage_rates_unret = numpy.dot(map_weights, cond_damage_rates_unret)
		damage_rates_ret = numpy.dot(map_weights, cond_damage_rates_ret)

		if numpy.all(numpy.greater(damage_rates_unret, numpy.zeros(damage_rates_unret.shape))) == True:
			sc = True

			with open(output_folder + 'U_good.pkl', 'wb') as f:
				pickle.dump(U, f)

			print 'found a good U.'

	# get bridge damage rates
	damage_rates = {}
	print '*** Damage rates for S = ', scenarios
	for b in bridge_ids:
		damage_rates[b] = {}
		damage_rates[b]['unret'] = damage_rates_unret[
			bridge_indices[b]]  # sum(damage_tracker_unret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		damage_rates[b]['ret'] = damage_rates_ret[
			bridge_indices[b]]  # sum(damage_tracker_ret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		print 'b = ', b, damage_rates[b]['unret'], damage_rates[b]['ret']

def iterate_damage_rates(cs_number, p, scenarios, dam_maps_per_scenario = 10):


	import os

	directory = 'sobol_output/'

	if not p:
		output_folder = directory + 'cs' + str(cs_number) + '/'
	else:
		output_folder = directory + 'cs' + str(cs_number) + 'p/'

	partial_dict = get_bridge_dict(cs_number)
	bridge_ids = get_bridge_ids(cs_number)

	n_bridges = len(bridge_ids)
	bridge_indices = {bridge_ids[i]: i for i in
					  range(0, len(bridge_ids))}  # each bridge has an index in the damage_tracker array

	# if scenarios == 25:
	map_indices_input = 'sobol_input/cs' + str(cs_number) + '_training_map_indices.pkl'  # S = 25 for cs1, cs2, cs3; S = 20 for cs9
	map_weights_input = 'sobol_input/cs' + str(cs_number) + '_training_map_weights.pkl'   # S = 25 for cs1, cs2, cs3; S = 20 for cs9

	with open(map_indices_input, 'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input, 'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

	print len(map_indices), map_indices
	print len(map_weights), map_weights

	assert len(
		map_indices) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map indices list.'
	assert len(
		map_weights) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map weights list.'

	# if os.path.isfile('debugging_sobol/U_sft2_subset_comps.pkl'):
	# 	with open('debugging_sobol/U_sft2_subset_comps.pkl', 'rb') as f:
	# 		U = pickle.load(f)
	# 		print 'loaded U from file'
	# else:
	# with open('debugging_sobol/U_sft2_subset_comps.pkl', 'wb') as f:
	# 	pickle.dump(U, f)

	# if scenarios == 20:
	# 	U_input = 'sobol_input/U_sft2_s20d10.pkl'
	# 	with open(U_input, 'rb') as f:
	# 		U = pickle.load(f)
	# elif scenarios == 40:
	# 	U = generate_uniform_numbers(n_bridges, scenarios*num_damage_maps, 'U_temp.pkl', store=False)
	# 	with open('debugging_sobol/U_sft2_s40_d10.pkl', 'wb') as f:
	# 		pickle.dump(U, f)

	# map_indices = [index//num_damage_maps for index in map_indices] # convert damage map indices to ground-motion map indices (none should be higher than 1992)
	# print 'converted map indices = ', map_indices
	# if U.shape[0] != scenarios * num_damage_maps:
	# 	# then select the subset of U pertaining to the subset being used
	# 	temp_U = numpy.zeros((scenarios*num_damage_maps, n_bridges))
	# 	i = 0
	# 	for s in map_indices:
	# 		temp_U[i*num_damage_maps:(i+1)*num_damage_maps,:] = U[s*num_damage_maps:(s+1)*num_damage_maps,:]
	# 		i += 1
	# 	U = temp_U # change pointer

	f_unretrofitted = numpy.zeros((1, n_bridges))
	f_retrofitted = numpy.zeros((1, n_bridges,))
	i = 0
	for b in bridge_ids:
		f_unretrofitted[0, i] = partial_dict[b]['ext_lnSa']
		f_retrofitted[0, i] = partial_dict[b]['ext_lnSa'] * partial_dict[b]['omega']
		i += 1

	print 'f_unretrofitted', f_unretrofitted
	print 'f_retrofitted', f_retrofitted

	# Set up traffic model and run it.
	G = mahmodel.get_graph()

	assert G.is_multigraph() == False, 'You want a directed graph without multiple edges between nodes'

	demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
							 'input/superdistricts_centroids_dummies.csv')

	no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = precompute_network_performance(G, demand)

	print 'no_damage_travel_time = ', no_damage_travel_time
	print 'no_damage_vmt = ', no_damage_vmt
	print 'no_damage_trips_made = ', no_damage_trips_made

	sc = False
	while sc is False:

		U_temp = generate_uniform_numbers(n_bridges, 1992 * dam_maps_per_scenario, 'U_temp.pkl', store=False)

		if scenarios < 1992:  # get subset of uniform random numbers that correspond to the scenarios of interest
			U = numpy.zeros((scenarios * dam_maps_per_scenario, n_bridges))
			i = 0
			for s in map_indices:
				U[i * dam_maps_per_scenario:(i + 1) * dam_maps_per_scenario, :] = U_temp[s * dam_maps_per_scenario:(
																															   s + 1) * dam_maps_per_scenario,
																				  :]
				i += 1

		assert U.shape[
				   0] == scenarios * dam_maps_per_scenario, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
		assert U.shape[
				   1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'

		assert U.shape[
				   0] == scenarios * dam_maps_per_scenario, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
		assert U.shape[
				   1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'

		damage_tracker_unret = numpy.zeros((scenarios * dam_maps_per_scenario, n_bridges, 1))  # array of size (SxD, B, 1)
		damage_tracker_ret = numpy.zeros((scenarios * dam_maps_per_scenario, n_bridges, 1))  # array of size (SxD, B, 1)

		damage_tracker_unret_output = output_folder + 'damage_tracker_unret.pkl'
		damage_tracker_ret_output = output_folder + 'damage_tracker_ret.pkl'

		damage_tracker_unret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights,
															  f_unretrofitted, U,
															  demand, damage_tracker_unret, bridge_indices,
															  no_damage_travel_time,
															  no_damage_vmt, no_damage_trips_made, num_gm_maps=scenarios,
															  num_damage_maps=dam_maps_per_scenario)

		damage_tracker_ret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights,
															f_retrofitted, U,
															demand, damage_tracker_ret, bridge_indices,
															no_damage_travel_time,
															no_damage_vmt, no_damage_trips_made, num_gm_maps=scenarios,
															num_damage_maps=dam_maps_per_scenario)

		with open(damage_tracker_unret_output, 'wb') as f:
			pickle.dump(damage_tracker_unret, f)
		with open(damage_tracker_ret_output, 'wb') as f:
			pickle.dump(damage_tracker_ret, f)

		print 'damage tracker shapes: ', damage_tracker_unret.shape, damage_tracker_ret.shape

		cond_damage_rates_unret = numpy.zeros((scenarios, n_bridges))
		cond_damage_rates_ret = numpy.zeros((scenarios, n_bridges))

		for s in range(0, scenarios):
			temp_unret = numpy.average(damage_tracker_unret[s * dam_maps_per_scenario:(s + 1) * dam_maps_per_scenario
									   , :], axis=0)
			temp_ret = numpy.average(damage_tracker_ret[s * dam_maps_per_scenario:(s + 1) * dam_maps_per_scenario, :], axis=0)
			cond_damage_rates_unret[s, :] = numpy.reshape(temp_unret, (n_bridges))
			cond_damage_rates_ret[s, :] = numpy.reshape(temp_ret, (n_bridges))

		damage_rates_unret = numpy.dot(map_weights, cond_damage_rates_unret)
		damage_rates_ret = numpy.dot(map_weights, cond_damage_rates_ret)

		print min(damage_rates_unret), max(damage_rates_unret),

		if numpy.all(numpy.greater(damage_rates_unret, numpy.zeros(damage_rates_unret.shape))) == True:# and numpy.all(numpy.less(damage_rates_unret, numpy.ones(damage_rates_unret.shape)*0.001)) == True:
			sc = True

			output_filename = 'U_good_cs' + str(cs_number) + '.pkl'
			with open(output_folder + output_filename, 'wb') as f:
				pickle.dump(U_temp, f)

			print 'found a good U.'

	# get bridge damage rates
	damage_rates = {}
	print '*** Damage rates for S = ', scenarios
	for b in bridge_ids:
		damage_rates[b] = {}
		damage_rates[b]['unret'] = damage_rates_unret[
			bridge_indices[b]]  # sum(damage_tracker_unret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		damage_rates[b]['ret'] = damage_rates_ret[
			bridge_indices[b]]  # sum(damage_tracker_ret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		print 'b = ', b, damage_rates[b]['unret'], damage_rates[b]['ret']

def iterate_damage_rates_sf_fullr(p, scenarios, dam_maps_per_scenario = 10):

	directory = 'sobol_output/'
	output_folder = directory + 'sf_fullr/'

	tag = 'sf_fullr'

	with open('input/sf_fullr_dict.pkl', 'rb') as f:
		partial_dict = pickle.load(f)

	with open('input/sf_fullr_bridge_ids.pkl', 'rb') as f:
		bridge_ids = pickle.load(f)

	n_bridges = len(bridge_ids)
	bridge_indices = {bridge_ids[i]: i for i in
					  range(0, len(bridge_ids))}  # each bridge has an index in the damage_tracker array

	map_indices_input = 'sobol_input/' + tag + '_training_map_indices.pkl'  # S = 30 for sf_fullr
	map_weights_input = 'sobol_input/' + tag + '_training_map_weights.pkl'   # S = 30 for sf_fullr

	with open(map_indices_input, 'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input, 'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

	print len(map_indices), map_indices
	print len(map_weights), map_weights

	assert len(
		map_indices) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map indices list.'
	assert len(
		map_weights) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map weights list.'

	# if os.path.isfile('debugging_sobol/U_sft2_subset_comps.pkl'):
	# 	with open('debugging_sobol/U_sft2_subset_comps.pkl', 'rb') as f:
	# 		U = pickle.load(f)
	# 		print 'loaded U from file'
	# else:
	# with open('debugging_sobol/U_sft2_subset_comps.pkl', 'wb') as f:
	# 	pickle.dump(U, f)

	# if scenarios == 20:
	# 	U_input = 'sobol_input/U_sft2_s20d10.pkl'
	# 	with open(U_input, 'rb') as f:
	# 		U = pickle.load(f)
	# elif scenarios == 40:
	# 	U = generate_uniform_numbers(n_bridges, scenarios*num_damage_maps, 'U_temp.pkl', store=False)
	# 	with open('debugging_sobol/U_sft2_s40_d10.pkl', 'wb') as f:
	# 		pickle.dump(U, f)

	# map_indices = [index//num_damage_maps for index in map_indices] # convert damage map indices to ground-motion map indices (none should be higher than 1992)
	# print 'converted map indices = ', map_indices
	# if U.shape[0] != scenarios * num_damage_maps:
	# 	# then select the subset of U pertaining to the subset being used
	# 	temp_U = numpy.zeros((scenarios*num_damage_maps, n_bridges))
	# 	i = 0
	# 	for s in map_indices:
	# 		temp_U[i*num_damage_maps:(i+1)*num_damage_maps,:] = U[s*num_damage_maps:(s+1)*num_damage_maps,:]
	# 		i += 1
	# 	U = temp_U # change pointer

	f_unretrofitted = numpy.zeros((1, n_bridges))
	f_retrofitted = numpy.zeros((1, n_bridges,))
	i = 0
	for b in bridge_ids:
		f_unretrofitted[0, i] = partial_dict[b]['ext_lnSa']
		f_retrofitted[0, i] = partial_dict[b]['ext_lnSa'] * partial_dict[b]['omega']
		i += 1

	print 'f_unretrofitted', f_unretrofitted
	print 'f_retrofitted', f_retrofitted

	# Set up traffic model and run it.
	G = mahmodel.get_graph()

	assert G.is_multigraph() == False, 'You want a directed graph without multiple edges between nodes'

	demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
							 'input/superdistricts_centroids_dummies.csv')

	no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = precompute_network_performance(G, demand)

	print 'no_damage_travel_time = ', no_damage_travel_time
	print 'no_damage_vmt = ', no_damage_vmt
	print 'no_damage_trips_made = ', no_damage_trips_made

	sc = False
	while sc is False:

		U_temp = generate_uniform_numbers(n_bridges, 1992 * dam_maps_per_scenario, 'U_temp.pkl', store=False)

		if scenarios < 1992:  # get subset of uniform random numbers that correspond to the scenarios of interest
			U = numpy.zeros((scenarios * dam_maps_per_scenario, n_bridges))
			i = 0
			for s in map_indices:
				U[i * dam_maps_per_scenario:(i + 1) * dam_maps_per_scenario, :] = U_temp[s * dam_maps_per_scenario:(
																															   s + 1) * dam_maps_per_scenario,
																				  :]
				i += 1

		assert U.shape[
				   0] == scenarios * dam_maps_per_scenario, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
		assert U.shape[
				   1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'

		assert U.shape[
				   0] == scenarios * dam_maps_per_scenario, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
		assert U.shape[
				   1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'

		damage_tracker_unret = numpy.zeros((scenarios * dam_maps_per_scenario, n_bridges, 1))  # array of size (SxD, B, 1)
		damage_tracker_ret = numpy.zeros((scenarios * dam_maps_per_scenario, n_bridges, 1))  # array of size (SxD, B, 1)

		damage_tracker_unret_output = output_folder + 'damage_tracker_unret.pkl'
		damage_tracker_ret_output = output_folder + 'damage_tracker_ret.pkl'

		damage_tracker_unret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights,
															  f_unretrofitted, U,
															  demand, damage_tracker_unret, bridge_indices,
															  no_damage_travel_time,
															  no_damage_vmt, no_damage_trips_made, num_gm_maps=scenarios,
															  num_damage_maps=dam_maps_per_scenario)

		damage_tracker_ret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights,
															f_retrofitted, U,
															demand, damage_tracker_ret, bridge_indices,
															no_damage_travel_time,
															no_damage_vmt, no_damage_trips_made, num_gm_maps=scenarios,
															num_damage_maps=dam_maps_per_scenario)

		with open(damage_tracker_unret_output, 'wb') as f:
			pickle.dump(damage_tracker_unret, f)
		with open(damage_tracker_ret_output, 'wb') as f:
			pickle.dump(damage_tracker_ret, f)

		print 'damage tracker shapes: ', damage_tracker_unret.shape, damage_tracker_ret.shape

		cond_damage_rates_unret = numpy.zeros((scenarios, n_bridges))
		cond_damage_rates_ret = numpy.zeros((scenarios, n_bridges))

		for s in range(0, scenarios):
			temp_unret = numpy.average(damage_tracker_unret[s * dam_maps_per_scenario:(s + 1) * dam_maps_per_scenario
									   , :], axis=0)
			temp_ret = numpy.average(damage_tracker_ret[s * dam_maps_per_scenario:(s + 1) * dam_maps_per_scenario, :], axis=0)
			cond_damage_rates_unret[s, :] = numpy.reshape(temp_unret, (n_bridges))
			cond_damage_rates_ret[s, :] = numpy.reshape(temp_ret, (n_bridges))

		damage_rates_unret = numpy.dot(map_weights, cond_damage_rates_unret)
		damage_rates_ret = numpy.dot(map_weights, cond_damage_rates_ret)

		print min(damage_rates_unret), max(damage_rates_unret),

		if numpy.all(numpy.greater(damage_rates_unret, numpy.zeros(damage_rates_unret.shape))) == True:# and numpy.all(numpy.less(damage_rates_unret, numpy.ones(damage_rates_unret.shape)*0.001)) == True:
			sc = True

			output_filename = 'U_good_' + tag + '.pkl'
			with open(output_folder + output_filename, 'wb') as f:
				pickle.dump(U_temp, f)

			print 'found a good U.'

	# get bridge damage rates
	damage_rates = {}
	print '*** Damage rates for S = ', scenarios
	for b in bridge_ids:
		damage_rates[b] = {}
		damage_rates[b]['unret'] = damage_rates_unret[
			bridge_indices[b]]  # sum(damage_tracker_unret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		damage_rates[b]['ret'] = damage_rates_ret[
			bridge_indices[b]]  # sum(damage_tracker_ret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		print 'b = ', b, damage_rates[b]['unret'], damage_rates[b]['ret']

def get_bridge_damage_rates_sf_fullr(p, dam_maps_per_scenario, scenarios, test= False):

	directory = 'sobol_output/'
	output_folder = directory + 'sf_fullr/'

	tag = 'sf_fullr'

	with open('input/sf_fullr_dict.pkl', 'rb') as f:
		partial_dict = pickle.load(f)

	with open('input/sf_fullr_bridge_ids.pkl', 'rb') as f:
		bridge_ids = pickle.load(f)
	n_bridges = len(bridge_ids)
	bridge_indices = {bridge_ids[i]:i for i in range(0,len(bridge_ids))} # each bridge has an index in the damage_tracker array

	if test:
		map_indices_input = 'sobol_input/' + tag + '_testing_map_indices.pkl'  # S = 25 for cs1, cs2, cs3; S = 20 for cs9
		map_weights_input = 'sobol_input/' + tag + '_testing_map_weights.pkl'  # S = 25 for cs1, cs2, cs3; S = 20 for cs9
	else:
		map_indices_input = 'sobol_input/' + tag +  '_training_map_indices.pkl'  # S = 25 for cs1, cs2, cs3; S = 20 for cs9
		map_weights_input = 'sobol_input/' + tag +  '_training_map_weights.pkl'  # S = 25 for cs1, cs2, cs3; S = 20 for cs9

	with open(map_indices_input,'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input,'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

	print len(map_indices), map_indices
	print len(map_weights), map_weights

	assert len(map_indices) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map indices list.'
	assert len(map_weights) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map weights list.'

	with open('sobol_output/' + tag + '/U_good_' + tag + '.pkl', 'rb') as f:
		U_temp = pickle.load(f)

	if scenarios < 1992:  # get subset of uniform random numbers that correspond to the scenarios of interest
		U = numpy.zeros((scenarios * dam_maps_per_scenario, n_bridges))
		i = 0
		for s in map_indices:
			U[i * dam_maps_per_scenario:(i + 1) * dam_maps_per_scenario, :] = U_temp[s * dam_maps_per_scenario:(s + 1) * dam_maps_per_scenario,
																  :]
			i += 1

	assert U.shape[0] == scenarios * dam_maps_per_scenario, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
	assert U.shape[1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'

	print 'U shapes (set and subset): ', U_temp.shape, U.shape
	damage_tracker_unret = numpy.zeros((scenarios*dam_maps_per_scenario,n_bridges, 1)) # array of size (SxD, B, 1)
	damage_tracker_ret = numpy.zeros((scenarios*dam_maps_per_scenario,n_bridges, 1)) # array of size (SxD, B, 1)

	# no_damage_travel_time = n_bridges

	f_unretrofitted = numpy.zeros((1,n_bridges))
	f_retrofitted = numpy.zeros((1,n_bridges,))
	i = 0
	for b in bridge_ids:
		f_unretrofitted[0,i] = partial_dict[b]['ext_lnSa']
		f_retrofitted[0, i] = partial_dict[b]['ext_lnSa'] * partial_dict[b]['omega']
		i += 1

	print 'f_unretrofitted', f_unretrofitted
	print 'f_retrofitted', f_retrofitted

	# # Set up traffic model and run it.
	# G = mahmodel.get_graph()
	#
	# assert G.is_multigraph() == False, 'You want a directed graph without multiple edges between nodes'
	#
	# demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
	# 						 'input/superdistricts_centroids_dummies.csv')
	#
	# no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = precompute_network_performance(G, demand)
	#
	# print 'no_damage_travel_time = ', no_damage_travel_time
	# print 'no_damage_vmt = ', no_damage_vmt
	# print 'no_damage_trips_made = ', no_damage_trips_made

	no_damage_travel_time = 0
	no_damage_vmt = 0
	no_damage_trips_made = 0
	demand = 0

	damage_tracker_unret_output = output_folder + 'damage_tracker_unret.pkl'
	damage_tracker_ret_output = output_folder + 'damage_tracker_ret.pkl'
	# if os.path.isfile(damage_tracker_unret_output):
	# 	with open(damage_tracker_unret_output, 'rb') as f:
	# 		damage_tracker_unret = pickle.load(f)
	# 	with open(damage_tracker_ret_output, 'rb') as f:
	# 		damage_tracker_ret = pickle.load(f)
	# else:
	damage_tracker_unret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights, f_unretrofitted, U,
														  demand, damage_tracker_unret, bridge_indices,no_damage_travel_time,
														  no_damage_vmt, no_damage_trips_made, num_gm_maps=scenarios,
															num_damage_maps=dam_maps_per_scenario)

	damage_tracker_ret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights, f_retrofitted, U,
														  demand, damage_tracker_ret, bridge_indices,no_damage_travel_time,
														  no_damage_vmt, no_damage_trips_made, num_gm_maps=scenarios,
															num_damage_maps=dam_maps_per_scenario)

	with open(damage_tracker_unret_output, 'wb') as f:
		pickle.dump(damage_tracker_unret, f)
	with open(damage_tracker_ret_output, 'wb') as f:
		pickle.dump(damage_tracker_ret, f)

	print 'damage tracker shapes: ', damage_tracker_unret.shape, damage_tracker_ret.shape

	cond_damage_rates_unret = numpy.zeros((scenarios,n_bridges))
	cond_damage_rates_ret = numpy.zeros((scenarios,n_bridges))

	for s in range(0, scenarios):
		temp_unret = numpy.average(damage_tracker_unret[s*dam_maps_per_scenario:(s+1)*dam_maps_per_scenario,:],axis=0)
		temp_ret = numpy.average(damage_tracker_ret[s*dam_maps_per_scenario:(s+1)*dam_maps_per_scenario,:],axis=0)
		cond_damage_rates_unret[s,:] = numpy.reshape(temp_unret,(n_bridges))
		cond_damage_rates_ret[s,:] =numpy.reshape(temp_ret,(n_bridges))

	damage_rates_unret = numpy.dot(map_weights, cond_damage_rates_unret)
	damage_rates_ret = numpy.dot(map_weights, cond_damage_rates_ret)

	damage_rates = {}
	print '*** Damage rates for S = ', scenarios
	dam_rates = []
	for b in bridge_ids:
		damage_rates[b] = {}
		damage_rates[b]['unret'] = damage_rates_unret[bridge_indices[b]]#sum(damage_tracker_unret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		damage_rates[b]['ret'] = damage_rates_ret[bridge_indices[b]]#sum(damage_tracker_ret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		dam_rates.append(damage_rates[b]['unret'])
		print 'b = ', b, damage_rates[b]['unret'], damage_rates[b]['ret']

	sc = False
	if numpy.count_nonzero(dam_rates) == n_bridges:
		sc = True

	return sc

def compare_dicts():

	partial_dict_name = 'sf_testbed_2_dict'
	partial_dict_path = 'input/' + partial_dict_name + '.pkl'
	with open(partial_dict_path, 'rb') as f:
		partial_dict = pickle.load(f)

	sf_testbed = [951, 1081, 935, 895, 947, 993, 976, 898, 925, 926, 966, 917, 1065, 953, 972, 990, 899, 919, 904,
					940]  # see bridge_metadata_NBI_sf_tetsbed/sf_testbed_new_3 -- otherwise referred to as sf_testbed_2
	bridge_ids = [str(b) for b in sf_testbed]

	new_partial_dict = create_partial_dict(bridge_ids)

	# for bridge in bridge_ids:
	# 	print 'bridge ', bridge
	# 	print 'old dict: ', partial_dict[bridge]
	# 	print 'new dict: ', new_partial_dict[bridge]
	# 	print 'before vs. after: ', len(partial_dict[bridge].keys()), len(new_partial_dict[bridge].keys())

	for bridge in bridge_ids:
		print bridge, 0.25*new_partial_dict[bridge]['area']*293/(28*10**6)

	with open('input/sft2_dict.pkl','wb') as f:
		pickle.dump(new_partial_dict,f)

def compare_proxy_metrics(scenarios, s2 = 1992, num_damage_maps = 10):

	if scenarios == 20:
		map_indices_input = 'sobol_input/sft2_training_map_indices.pkl'  # S = 20
		map_weights_input = 'sobol_input/sft2_training_map_weights.pkl'  # S = 20
		output_folder = 'debugging_sobol/s20/'
	elif scenarios == 38:
		map_indices_input = 'sobol_input/sft2_testing_map_indices.pkl'  # S = 38
		map_weights_input = 'sobol_input/sft2_testing_map_weights.pkl'  # S = 38
		output_folder = 'debugging_sobol/s38/'
	elif scenarios == 15:
		map_indices_input = 'sobol_input/sft2_k15_map_indices.pkl'
		map_weights_input = 'sobol_input/sft2_k15_map_weights.pkl'
		output_folder = 'debugging_sobol/s15/'
	else:
		print 'ERROR: scenarios should be 15, 20, or 38.'

	if s2 == 1992:
		map_indices_input_2 = 'sobol_input/ucerf2_map_indices.pkl'
		map_weights_input_2 = 'sobol_input/ucerf2_map_weights.pkl'
		output_folder_2 = 'debugging_sobol/s1992/'


	# LOAD FIRST SET (S1)
	with open(map_indices_input,'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input,'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

	# LOAD SECOND SET (S2) FOR COMPARISON
	with open(map_indices_input_2,'rb') as f:
		map_indices_2 = pickle.load(f)

	with open(map_weights_input_2,'rb') as f:
		map_weights_2 = pickle.load(f)

	if len(map_indices_2) != s2:
		map_indices_2 = map_indices_2[0]
		map_weights_2 = map_weights_2[0]

	# get fraction of bridges damaged in each damage map for s1 and s2
	import os
	damage_tracker_unret_output = output_folder + 'damage_tracker_unret.pkl'
	if os.path.isfile(damage_tracker_unret_output):
		with open(damage_tracker_unret_output, 'rb') as f:
			damage_tracker_unret = pickle.load(f)

	s2_damage_tracker_output = output_folder_2+ 'damage_tracker_unret.pkl'
	with open(s2_damage_tracker_output, 'rb') as f:
		damage_tracker_unret_s2 = pickle.load(f)

	cond_damage_fractions_unret = numpy.zeros((scenarios*num_damage_maps,))
	cond_damage_fractions_s2 = numpy.zeros((s2*num_damage_maps,))
	cond_damage = numpy.zeros((scenarios,))
	cond_damage_2 = numpy.zeros((s2,))

	for s in range(0, scenarios):
		temp_unret = numpy.average(damage_tracker_unret[s*num_damage_maps:(s+1)*num_damage_maps,:], axis = 1) # get fraction of bridges damaged in each damage map for scenario s
		cond_damage_fractions_unret[s*num_damage_maps:(s+1)*num_damage_maps] = numpy.reshape(temp_unret, (num_damage_maps,))
		cond_damage[s] = numpy.average(temp_unret)

	for s in range(0,s2):
		temp_unret_2 = numpy.average(damage_tracker_unret_s2[s * num_damage_maps:(s + 1) * num_damage_maps, :], axis=1)
		cond_damage_fractions_s2[s * num_damage_maps:(s + 1) * num_damage_maps] = numpy.reshape(temp_unret_2, (num_damage_maps,))
		cond_damage_2[s] = numpy.average(temp_unret_2)

	print 's1 = ', numpy.dot(map_weights, cond_damage)
	print 's2 = ', numpy.dot(map_weights_2, cond_damage_2)

def compare_map_weights(s1, s2 = 1992):

	if s1 == 30:
		map_indices_input = 'sobol_input/sf_full_s30_map_indices.pkl'  # S = 30
		map_weights_input = 'sobol_input/sf_full_s30_training_map_weights.pkl'  # S = 30

	if s2 == 1992:
		map_indices_input_2 = 'sobol_input/ucerf2_map_indices.pkl'
		map_weights_input_2 = 'sobol_input/ucerf2_map_weights.pkl'

	# LOAD FIRST SET (S1)
	with open(map_indices_input,'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input,'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != s1:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

	# LOAD SECOND SET (S2) FOR COMPARISON
	with open(map_indices_input_2,'rb') as f:
		map_indices_2 = pickle.load(f)

	with open(map_weights_input_2,'rb') as f:
		map_weights_2 = pickle.load(f)

	if len(map_indices_2) != s2:
		map_indices_2 = map_indices_2[0]
		map_weights_2 = map_weights_2[0]

	i = 0
	for s in map_indices:
		s2_index = map_indices_2.index(s)
		print s, map_weights[i], map_weights_2[s2_index], map_weights[i]/map_weights_2[s2_index]
		i += 1

def graph_bridge_repair_costs():


	import operator

	partial_dict_name = 'sft2_dict'  # sf_dict if sf_full; previously 'sf_testbed_2_dict' but that didn't include bridge areas
	partial_dict_path = 'input/' + partial_dict_name + '.pkl'
	with open(partial_dict_path, 'rb') as f:
		partial_dict = pickle.load(f)
	sf_testbed = [951, 1081, 935, 895, 947, 993, 976, 898, 925, 926, 966, 917, 1065, 953, 972, 990, 899, 919, 904,
				  940]  # see bridge_metadata_NBI_sf_tetsbed/sf_testbed_new_3 -- otherwise referred to as sf_testbed_2
	bridge_ids = [str(b) for b in sf_testbed]
	n_bridges = len(bridge_ids)

	repair_cost = {}

	for bridge in bridge_ids:
		repair_cost[bridge] = partial_dict[bridge]['area']*293*0.25


	sorted_repair_cost = sorted(repair_cost.items(), key=operator.itemgetter(1), reverse=True)
	repair_cost_bridge_id = []
	repair_cost_list = []
	repair_cost_index = []
	for i in range(0, n_bridges):
		repair_cost_bridge_id.append(sorted_repair_cost[i][0])
		repair_cost_list.append(sorted_repair_cost[i][1])
		repair_cost_index.append(i)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.bar(repair_cost_index, repair_cost_list, align='center')
	ax.set_yscale('log')
	ax.set_xticks(numpy.arange(0, n_bridges, step=1))
	ax.set_xticklabels(repair_cost_bridge_id, rotation=45, ha='right')
	ax.set_xlabel('Bridge ID (original)')
	ax.set_ylabel('Estimated repair cost for extensive damage')
	plt.savefig('sft2_bridge_repair_costs', bbox_inches='tight')

def get_bridge_damage_rates(cs_number, p, dam_maps_per_scenario, scenarios, test= False):

	directory = 'sobol_output/'

	if not p:
		output_folder = directory + 'cs' + str(cs_number) + '/'
	else:
		output_folder = directory + 'cs' + str(cs_number) + 'p/'

	partial_dict = get_bridge_dict(cs_number)
	bridge_ids = get_bridge_ids(cs_number)

	n_bridges = len(bridge_ids)
	bridge_indices = {bridge_ids[i]:i for i in range(0,len(bridge_ids))} # each bridge has an index in the damage_tracker array

	# if scenarios == 25:
	if test:
		map_indices_input = 'sobol_input/cs' + str(
			cs_number) + '_testing_map_indices.pkl'  # S = 25 for cs1, cs2, cs3; S = 20 for cs9
		map_weights_input = 'sobol_input/cs' + str(
			cs_number) + '_testing_map_weights.pkl'  # S = 25 for cs1, cs2, cs3; S = 20 for cs9
	else:
		map_indices_input = 'sobol_input/cs' + str(cs_number) + '_training_map_indices.pkl'  # S = 25 for cs1, cs2, cs3; S = 20 for cs9
		map_weights_input = 'sobol_input/cs' + str(cs_number) + '_training_map_weights.pkl'  # S = 25 for cs1, cs2, cs3; S = 20 for cs9

	with open(map_indices_input,'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input,'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

	print len(map_indices), map_indices
	print len(map_weights), map_weights

	assert len(map_indices) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map indices list.'
	assert len(map_weights) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map weights list.'

	# U = generate_uniform_numbers(n_bridges, scenarios*num_damage_maps, 'U_temp.pkl', store=False)

	# U = generate_uniform_numbers(n_bridges, scenarios*num_damage_maps, 'U_temp.pkl', store = False)

	# with open('sobol_input/U_cs_s1992d10.pkl', 'rb') as f:
	# 	U_temp = pickle.load(f)

	with open('sobol_output/cs1/U_good.pkl', 'rb') as f:
		U_temp = pickle.load(f)

	if scenarios < 1992:  # get subset of uniform random numbers that correspond to the scenarios of interest
		U = numpy.zeros((scenarios * dam_maps_per_scenario, n_bridges))
		i = 0
		for s in map_indices:
			U[i * dam_maps_per_scenario:(i + 1) * dam_maps_per_scenario, :] = U_temp[s * dam_maps_per_scenario:(s + 1) * dam_maps_per_scenario,
																  :]
			i += 1

	assert U.shape[0] == scenarios * dam_maps_per_scenario, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
	assert U.shape[1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'

	print 'U shapes (set and subset): ', U_temp.shape, U.shape
	damage_tracker_unret = numpy.zeros((scenarios*dam_maps_per_scenario,n_bridges, 1)) # array of size (SxD, B, 1)
	damage_tracker_ret = numpy.zeros((scenarios*dam_maps_per_scenario,n_bridges, 1)) # array of size (SxD, B, 1)

	# no_damage_travel_time = n_bridges

	f_unretrofitted = numpy.zeros((1,n_bridges))
	f_retrofitted = numpy.zeros((1,n_bridges,))
	i = 0
	for b in bridge_ids:
		f_unretrofitted[0,i] = partial_dict[b]['ext_lnSa']
		f_retrofitted[0, i] = partial_dict[b]['ext_lnSa'] * partial_dict[b]['omega']
		i += 1

	print 'f_unretrofitted', f_unretrofitted
	print 'f_retrofitted', f_retrofitted

	# # Set up traffic model and run it.
	# G = mahmodel.get_graph()
	#
	# assert G.is_multigraph() == False, 'You want a directed graph without multiple edges between nodes'
	#
	# demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
	# 						 'input/superdistricts_centroids_dummies.csv')
	#
	# no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = precompute_network_performance(G, demand)
	#
	# print 'no_damage_travel_time = ', no_damage_travel_time
	# print 'no_damage_vmt = ', no_damage_vmt
	# print 'no_damage_trips_made = ', no_damage_trips_made

	no_damage_travel_time = 0
	no_damage_vmt = 0
	no_damage_trips_made = 0
	demand = 0

	damage_tracker_unret_output = output_folder + 'damage_tracker_unret.pkl'
	damage_tracker_ret_output = output_folder + 'damage_tracker_ret.pkl'
	# if os.path.isfile(damage_tracker_unret_output):
	# 	with open(damage_tracker_unret_output, 'rb') as f:
	# 		damage_tracker_unret = pickle.load(f)
	# 	with open(damage_tracker_ret_output, 'rb') as f:
	# 		damage_tracker_ret = pickle.load(f)
	# else:
	damage_tracker_unret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights, f_unretrofitted, U,
														  demand, damage_tracker_unret, bridge_indices,no_damage_travel_time,
														  no_damage_vmt, no_damage_trips_made, num_gm_maps=scenarios,
															num_damage_maps=dam_maps_per_scenario)

	damage_tracker_ret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights, f_retrofitted, U,
														  demand, damage_tracker_ret, bridge_indices,no_damage_travel_time,
														  no_damage_vmt, no_damage_trips_made, num_gm_maps=scenarios,
															num_damage_maps=dam_maps_per_scenario)

	with open(damage_tracker_unret_output, 'wb') as f:
		pickle.dump(damage_tracker_unret, f)
	with open(damage_tracker_ret_output, 'wb') as f:
		pickle.dump(damage_tracker_ret, f)

	print 'damage tracker shapes: ', damage_tracker_unret.shape, damage_tracker_ret.shape

	cond_damage_rates_unret = numpy.zeros((scenarios,n_bridges))
	cond_damage_rates_ret = numpy.zeros((scenarios,n_bridges))

	for s in range(0, scenarios):
		temp_unret = numpy.average(damage_tracker_unret[s*dam_maps_per_scenario:(s+1)*dam_maps_per_scenario,:],axis=0)
		temp_ret = numpy.average(damage_tracker_ret[s*dam_maps_per_scenario:(s+1)*dam_maps_per_scenario,:],axis=0)
		cond_damage_rates_unret[s,:] = numpy.reshape(temp_unret,(n_bridges))
		cond_damage_rates_ret[s,:] =numpy.reshape(temp_ret,(n_bridges))

	damage_rates_unret = numpy.dot(map_weights, cond_damage_rates_unret)
	damage_rates_ret = numpy.dot(map_weights, cond_damage_rates_ret)

	damage_rates = {}
	print '*** Damage rates for S = ', scenarios
	dam_rates = []
	for b in bridge_ids:
		damage_rates[b] = {}
		damage_rates[b]['unret'] = damage_rates_unret[bridge_indices[b]]#sum(damage_tracker_unret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		damage_rates[b]['ret'] = damage_rates_ret[bridge_indices[b]]#sum(damage_tracker_ret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		dam_rates.append(damage_rates[b]['unret'])
		print 'b = ', b, damage_rates[b]['unret'], damage_rates[b]['ret']

	sc = False
	if numpy.count_nonzero(dam_rates) == n_bridges:
		sc = True

	return sc

def compare_bridge_damage_rates(num_damage_maps, scenarios = 30, test = False):

	output_folder = 'debugging_sobol/s30/'

	partial_dict_name = 'sf_full_dict'
	# load bridge dict, fragility function parameter samples, and uniform random number samples
	partial_dict_path = 'input/' + partial_dict_name + '.pkl'
	with open(partial_dict_path, 'rb') as f:
		partial_dict = pickle.load(f)

	with open('input/sf_full_bridge_ids.pkl','rb') as f:
		bridge_ids = pickle.load(f)

	print 'bridges = ', bridge_ids

	n_bridges = len(bridge_ids)
	bridge_indices = {bridge_ids[i]: i for i in
					  range(0, len(bridge_ids))}  # each bridge has an index in the damage_tracker array

	if scenarios == 30:
		map_indices_input = 'sobol_input/sf_full_s30_map_indices.pkl'  # S = 30
		map_weights_input = 'sobol_input/sf_full_s30_map_weights.pkl'  # S = 30

	comp_map_indices_input = 'sobol_input/ucerf2_map_indices.pkl'
	comp_map_weights_input = 'sobol_input/ucerf2_map_weights.pkl'
	comp_output_folder = 'debugging_sobol/s1992/'

	with open(map_indices_input, 'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input, 'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

	print len(map_indices), map_indices
	print len(map_weights), map_weights

	assert len(
		map_indices) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map indices list.'
	assert len(
		map_weights) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map weights list.'

	with open(comp_map_indices_input,'rb') as f:
		comp_map_indices = pickle.load(f)

	with open(comp_map_weights_input,'rb') as f:
		comp_map_weights = pickle.load(f)

	if len(comp_map_indices) != 1992:
		comp_map_indices = map_indices[0]
		comp_map_weights = map_weights[0]

	assert len(
		comp_map_indices) == 1992, 'Error: The number of scenarios (S) does not match the length of the map indices list.'
	assert len(
		comp_map_weights) == 1992, 'Error: The number of scenarios (S) does not match the length of the map weights list.'

	# with open('sobol_input/U_temp.pkl', 'rb') as f:
	# 	U_temp = pickle.load(f)

	U_temp = generate_uniform_numbers(n_bridges, 1992*num_damage_maps, 'U_temp', store=False)

	if scenarios < 1992:  # get subset of uniform random numbers that correspond to the scenarios of interest
		U = numpy.zeros((scenarios * num_damage_maps, n_bridges))
		i = 0
		for s in map_indices:
			U[i * num_damage_maps:(i + 1) * num_damage_maps, :] = U_temp[s * num_damage_maps:(s + 1) * num_damage_maps,
																  :]
			i += 1

	assert U.shape[
			   0] == scenarios * num_damage_maps, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
	assert U.shape[
			   1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'


	print 'U shapes (set and subset): ', U_temp.shape, U.shape

	f_unretrofitted = numpy.zeros((1, n_bridges))
	f_retrofitted = numpy.zeros((1, n_bridges,))
	i = 0
	for b in bridge_ids:
		f_unretrofitted[0, i] = partial_dict[b]['ext_lnSa']
		f_retrofitted[0, i] = partial_dict[b]['ext_lnSa'] * partial_dict[b]['omega']
		i += 1

	no_damage_travel_time = 0
	no_damage_vmt = 0
	no_damage_trips_made = 0
	demand = 0

	damage_tracker_unret_output = output_folder + 'damage_tracker_unret.pkl'
	damage_tracker_ret_output = output_folder + 'damage_tracker_ret.pkl'

	comp_damage_tracker_unret_output = comp_output_folder +  'damage_tracker_unret.pkl'
	comp_damage_tracker_ret_output = comp_output_folder +  'damage_tracker_ret.pkl'


	# damage_tracker_unret = numpy.zeros((num_damage_maps, n_bridges, 1))  # array of size (SxD, B, 1)
	# damage_tracker_ret = numpy.zeros((num_damage_maps, n_bridges, 1))  # array of size (SxD, B, 1)

	damage_tracker_unret = numpy.zeros((scenarios * num_damage_maps, n_bridges, 1))  # array of size (SxD, B, 1)
	damage_tracker_ret = numpy.zeros((scenarios * num_damage_maps, n_bridges, 1))  # array of size (SxD, B, 1)

	comp_damage_tracker_unret = numpy.zeros((1992*num_damage_maps, n_bridges, 1))
	comp_damage_tracker_ret = numpy.zeros((1992*num_damage_maps, n_bridges, 1))

	damage_tracker_unret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights,
														  f_unretrofitted, U,
														  demand, damage_tracker_unret, bridge_indices,
														  no_damage_travel_time,
														  no_damage_vmt, no_damage_trips_made, num_gm_maps=10,
														  num_damage_maps=num_damage_maps)

	damage_tracker_ret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, map_indices, map_weights,
														f_retrofitted, U,
														demand, damage_tracker_ret, bridge_indices,
														no_damage_travel_time,
														no_damage_vmt, no_damage_trips_made, num_gm_maps=scenarios,
														num_damage_maps=num_damage_maps)

	# just run the damage module for the maps in the subset

	comp_damage_tracker_unret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, comp_map_indices, comp_map_weights,
														  f_unretrofitted, U_temp,
														  demand, comp_damage_tracker_unret, bridge_indices,
														  no_damage_travel_time,
														  no_damage_vmt, no_damage_trips_made, num_gm_maps=1992,
														  num_damage_maps=num_damage_maps)

	comp_damage_tracker_ret = run_traffic_model_set_dam_only(bridge_ids, partial_dict, comp_map_indices, comp_map_weights,
														f_retrofitted, U_temp,
														demand, comp_damage_tracker_ret, bridge_indices,
														no_damage_travel_time,
														no_damage_vmt, no_damage_trips_made, num_gm_maps=1992,
														num_damage_maps=num_damage_maps)

	with open(damage_tracker_unret_output, 'wb') as f:
		pickle.dump(damage_tracker_unret, f)
	with open(damage_tracker_ret_output, 'wb') as f:
		pickle.dump(damage_tracker_ret, f)

	with open(comp_damage_tracker_unret_output, 'wb') as f:
		pickle.dump(comp_damage_tracker_unret, f)
	with open(comp_damage_tracker_ret_output, 'wb') as f:
		pickle.dump(comp_damage_tracker_ret, f)

	with open(damage_tracker_unret_output, 'rb') as f:
		damage_tracker_unret = pickle.load(f)
	with open(damage_tracker_ret_output, 'rb') as f:
		damage_tracker_ret = pickle.load(f)

	with open(comp_damage_tracker_unret_output, 'rb') as f:
		comp_damage_tracker_unret = pickle.load(f)
	with open(comp_damage_tracker_ret_output, 'rb') as f:
		comp_damage_tracker_ret = pickle.load(f)

	print 'damage tracker shapes: ', damage_tracker_unret.shape, damage_tracker_ret.shape, comp_damage_tracker_unret.shape, comp_damage_tracker_ret.shape

	cond_damage_rates_unret = numpy.zeros((scenarios, n_bridges))
	cond_damage_rates_ret = numpy.zeros((scenarios, n_bridges))
	comp_cond_damage_rates_unret = numpy.zeros((1992, n_bridges))
	comp_cond_damage_rates_ret = numpy.zeros((1992, n_bridges))

	for s in range(0, scenarios):
		temp_unret = numpy.average(damage_tracker_unret[s * num_damage_maps:(s + 1) * num_damage_maps, :], axis=0)
		temp_ret = numpy.average(damage_tracker_ret[s * num_damage_maps:(s + 1) * num_damage_maps, :], axis=0)
		cond_damage_rates_unret[s, :] = numpy.reshape(temp_unret, (n_bridges))
		cond_damage_rates_ret[s, :] = numpy.reshape(temp_ret, (n_bridges))

	for s in range(0, 1992):
		temp_unret = numpy.average(comp_damage_tracker_unret[s * num_damage_maps:(s + 1) * num_damage_maps, :], axis=0)
		temp_ret = numpy.average(comp_damage_tracker_ret[s * num_damage_maps:(s + 1) * num_damage_maps, :], axis=0)
		comp_cond_damage_rates_unret[s, :] = numpy.reshape(temp_unret, (n_bridges))
		comp_cond_damage_rates_ret[s, :] = numpy.reshape(temp_ret, (n_bridges))

	damage_rates_unret = numpy.dot(map_weights, cond_damage_rates_unret)
	damage_rates_ret = numpy.dot(map_weights, cond_damage_rates_ret)
	comp_damage_rates_unret = numpy.dot(comp_map_weights, comp_cond_damage_rates_unret)
	comp_damage_rates_ret = numpy.dot(comp_map_weights, comp_cond_damage_rates_ret)


	# get bridge damage rates
	print '*** Conditional damage rates for one of S = ', scenarios
	s_temp = 0 # scenario index in subset
	comp_s_temp = comp_map_indices.index(map_indices[s_temp]) # scenario index in full set

	print '*** Damage tracker for subset', s_temp, damage_tracker_unret.shape
	print damage_tracker_unret[s_temp*num_damage_maps:(s_temp+1)*num_damage_maps, :, 0]
	#
	# print '*** U for subset ', s_temp
	# print U[s_temp*num_damage_maps:(s_temp+1)*num_damage_maps, :]

	print '*** Damage tracker for full set', comp_s_temp, comp_damage_tracker_unret.shape
	print comp_damage_tracker_unret[comp_s_temp * num_damage_maps:(comp_s_temp + 1) * num_damage_maps, :, 0]

	# print '*** U for full set ', comp_s_temp
	# print U_temp[comp_s_temp * num_damage_maps:(comp_s_temp + 1) * num_damage_maps, :]

	print '*** Are damage trackers equal? ', numpy.array_equal(damage_tracker_unret[s_temp*num_damage_maps:(s_temp+1)*num_damage_maps, :, 0], comp_damage_tracker_unret[comp_s_temp * num_damage_maps:(comp_s_temp + 1) * num_damage_maps, :, 0])

	# print U[s_temp,:]
	# print U_temp[comp_s_temp*10,:]

	for b in bridge_ids: # the two columns should match
		print 'b = ', b, cond_damage_rates_unret[s_temp, bridge_indices[b]], comp_cond_damage_rates_unret[comp_s_temp, bridge_indices[b]]

	damage_rates = {}
	print '*** Comparing damage rates for S = ', scenarios
	for b in bridge_ids:
		damage_rates[b] = {}
		damage_rates[b]['unret'] = damage_rates_unret[
			bridge_indices[b]]  # sum(damage_tracker_unret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		damage_rates[b]['ret'] = damage_rates_ret[
			bridge_indices[b]]  # sum(damage_tracker_ret[:][bridge_indices[b]])/(scenarios*num_damage_maps)
		# print 'b = ', b, damage_rates[b]['unret'], damage_rates[b]['ret']
		print 'b = ', b, damage_rates[b]['unret'], comp_damage_rates_unret[bridge_indices[b]]

# --------------------------------------------- SOBOL INDEX METHODS ----------------------------------------------------
def interleave(x, z, u_x, u_z):
	# interleave two vectors x and z given u and u' to produce a hybrid point
	# takes as input vectors x and z, which should have the same shape, and indices u_x (elements to get
	# from x) and indices u_z (elements to get from z)
	# returns a single vector v, which is x and z interleaved -- it's a numpy array
	# x_sub = [x[i] for i in u_x] # select the appropriate values of the vector x
	# z_sub = [z[i] for i in u_z] # select the appropriate values of the vector z
	# v = numpy.concatenate((x_sub,z_sub)) # create the interleaved vector

	n_dims = x.shape[0]

	assert len(u_x) + len(u_z) == n_dims, 'Error in interleaving inputs'

	v = numpy.zeros((n_dims,))

	for i in range(0,n_dims):
		if i in u_x:
			v[i] = x[i]
		elif i in u_z:
			v[i] = z[i]

	return v

def interleave_set(X, Z, first_order=False):
	# interleave a set of vectors for every combination of u and u' to produce a set of hybrid points
	# takes as input:
	# X, an array of size (n_samples, n_dims)
	# Z, an array of size (n_samples, n_dims)
	# first, a boolean indicating whether interleaving should be done for the first-order Sobol index (True) or
	# total-order Sobol index (False) -- default is False
	# returns V, an array of size (n_samples, n_dims, n_dims) where each row is a hybrid point vector
	# and each page is associated with a particular dimension (bridge)

	n_samples = X.shape[0]
	n_dims = X.shape[1]

	assert X.shape == Z.shape, 'Error in dimensions of X and Z -- cannot be interleaved.'

	V = numpy.empty((n_samples, n_dims, n_dims))

	u = numpy.asarray(range(0,n_dims)) # all indices


	if first_order:
		for i in range(0, n_dims):

			u_z = [u[j] for j in u if j != i]  # u'
			u_x = numpy.asarray(u[i])
			u_x = numpy.reshape(u_x, (1,))  # u


			for k in range(0, n_samples):
				V[k, :, i] = interleave(X[k], Z[k], u_x, u_z)
	else:
		for i in range(0, n_dims):

			u_x = [u[j] for j in u if j != i] # u'
			u_z = numpy.asarray(u[i])
			u_z = numpy.reshape(u_z,(1,)) # u

			for k in range(0,n_samples):
				V[k, :, i] = interleave(X[k],Z[k],u_x,u_z)

	return V

def precompute_network_performance(): # copied from compute_bridge_sobol.py but modified to be county-specific

	G = mahmodel.get_graph()
	demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
							 'input/superdistricts_centroids_dummies.csv')
	no_damage_tt, no_damage_vmt, no_damage_trips_made = mahmodel.compute_tt_vmt(G, demand)

	G = util.clean_up_graph(G)  # so the trips assigned don't hang around

	return no_damage_tt, no_damage_vmt, no_damage_trips_made, G

# def run_traffic_model(bridge_dict, x, U, num_gm_maps = 10, num_damage_maps = 3):
# 	# run Miller's full traffic model (damage and traffic) on a set of specific bridges with particular fragilities
# 	# takes as input:
# 	# bridge_ids, a list of original bridge_ids that can be used as keys for partial_napa_dict
# 	# x, a numpy array of size (1, n_bridges) in which each row is a set of sampled bridge fragility parameters
# 	# U, a numpy array of size (num_gm_maps, n_bridges) in which each element is a sample from the standard uniform distribution
# 	# num_gm_maps, the number of earthquake scenarios to consider (of the 25 available)
# 	# num_damage_maps, the number of damage maps to create per scenario
# 	# returns the output of mahmodel_road_only_napa.py for an input of x
#
# 	start = time.time()
#
# 	# create the MTC highway graph
# 	G = mahmodel.get_graph()
# 	# compute the network performance measures of interest when there is no damage
# 	no_damage_travel_time, no_damage_vmt, no_damage_flow, no_damage_shortest_path, no_damage_lost_flow, G = precompute_network_performance(G)
#
# 	print 'no damage travel time = ', no_damage_travel_time
# 	print 'no damage lost flow = ', no_damage_lost_flow
#
# 	# run the traffic model for the one set of fragility parameters in x
# 	travel_times, average_travel_time, traffic_time, damage_tracker, delay_cost, connectivity_cost, indirect_cost = mahmodel.main(bridge_dict, x, U, G, no_damage_travel_time, no_damage_vmt, no_damage_flow,
# 								 no_damage_shortest_path, no_damage_lost_flow, num_gm_maps = num_gm_maps, num_damage_maps = num_damage_maps)
#
# 	print 'running the traffic model on 1 set of fragility parameters with ', num_damage_maps*num_gm_maps, ' damage maps took ', time.time() - start, ' seconds'
#
# 	return average_travel_time, indirect_cost

def run_traffic_model_set(bridge_ids, bridge_dict, map_indices, map_weights, X, U, demand, damage_tracker, bridge_indices,
						  no_damage_travel_time, no_damage_vmt, no_damage_trips_made, num_gm_maps = 10, num_damage_maps = 3, ret_cost = False):
	# def main(napa_dict, x, U, num_gm_maps = 10, num_damage_maps = 3):
	# run Miller's full traffic model (damage and traffic) on a set of specific bridges with various sets of fragilities
	# takes as input:
	# bridge_ids, a list of original bridge_ids that can be used as keys for partial_napa_dict
	# X, a numpy array of size (n_samples, n_bridges) in which each row is a set of sampled bridge fragility parameters
	# U, a numpy array of size (num_gm_maps, n_bridges) in which each element is a sample from the standard uniform distribution
	# num_gm_maps, the number of earthquake scenarios to consider (of the 25 available)
	# num_damage_maps, the number of damage maps to create per scenario
	# returns f_X, a numpy array of size (n_samples,) in which each element is the output of mahmodel_road_only_napa.py

	n_samples = X.shape[0]

	print 'n_samples = ', n_samples

	f_X_times = numpy.zeros((n_samples, num_gm_maps*num_damage_maps))
	f_X_trips = numpy.zeros((n_samples, num_gm_maps*num_damage_maps))
	f_X_vmts = numpy.zeros((n_samples, num_gm_maps*num_damage_maps))
	f_X_delay_costs = numpy.zeros((n_samples, num_gm_maps*num_damage_maps))
	f_X_conn_costs = numpy.zeros((n_samples, num_gm_maps*num_damage_maps))
	f_X_indirect_costs = numpy.zeros((n_samples, num_gm_maps*num_damage_maps))
	f_X_direct_costs = numpy.zeros((n_samples, num_gm_maps*num_damage_maps))

	f_X_avg_time = numpy.zeros((n_samples,))
	f_X_avg_trip = numpy.zeros((n_samples,))
	f_X_exp_indirect_cost = numpy.zeros((n_samples,))
	f_X_exp_direct_cost = numpy.zeros((n_samples,))
	f_X_exp_cost = numpy.zeros((n_samples,)) # total expected cost
	f_X_avg_vmt = numpy.zeros((n_samples,))
	f_X_ret_cost = numpy.zeros((n_samples,))

	# run the traffic model on each set of sampled fragility parameters and store the result of each run
	start = time.time()

	for i in range(0,n_samples):

		tt, trips, vmts, average_tt, average_trips, average_vmt, temp_damage_tracker, delay_costs, \
		connectivity_costs, indirect_costs, repair_costs, expected_delay_cost, expected_conn_cost, \
		expected_indirect_cost, expected_repair_cost, expected_total_cost, retrofit_cost = mahmodel.main(i, map_indices, map_weights, bridge_ids,bridge_dict, X[i,:],
												   U, demand, damage_tracker, bridge_indices,
												   no_damage_travel_time, no_damage_vmt, no_damage_trips_made,
												   num_gm_maps, num_damage_maps, ret_cost)

		# save raw performance data
		f_X_times[i,:] = tt
		f_X_trips[i,:] = trips
		f_X_vmts[i,:] = vmts

		# save raw cost data
		f_X_delay_costs[i,:] = delay_costs
		f_X_conn_costs[i,:] = connectivity_costs
		f_X_indirect_costs[i, :] = indirect_costs
		f_X_direct_costs[i,:] = repair_costs

		# save expected data
		f_X_avg_time[i] = average_tt
		f_X_avg_trip[i] = average_trips
		f_X_avg_vmt[i] = average_vmt
		f_X_exp_indirect_cost[i] = expected_indirect_cost
		f_X_exp_direct_cost[i] = expected_repair_cost
		f_X_exp_cost[i] = expected_total_cost
		f_X_ret_cost[i] = retrofit_cost

		damage_tracker = temp_damage_tracker

		print 'done with mahmodel for sample ', i + 1 , 'of ', n_samples, ' len(f_X) = ', f_X_times.shape[0]

	end = time.time()
	print 'run_traffic_model_set took ', end-start, ' seconds for ', n_samples, ' samples.'

	# return f_X_time, f_X_flows, f_X_vmts, f_X_cost, damage_tracker
	return f_X_times, f_X_trips, f_X_vmts, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, f_X_direct_costs, \
		   f_X_avg_time, f_X_avg_trip, f_X_avg_vmt, f_X_exp_indirect_cost, f_X_exp_direct_cost, f_X_exp_cost, f_X_ret_cost, damage_tracker

def run_traffic_model_set_dam_only(bridge_ids, bridge_dict, map_indices, map_weights, X, U, demand, damage_tracker, bridge_indices,
						  no_damage_travel_time, no_damage_vmt, no_damage_trips_made, num_gm_maps = 10, num_damage_maps = 3):
	# def main(napa_dict, x, U, num_gm_maps = 10, num_damage_maps = 3):
	# run Miller's full traffic model (damage and traffic) on a set of specific bridges with various sets of fragilities
	# takes as input:
	# bridge_ids, a list of original bridge_ids that can be used as keys for partial_napa_dict
	# X, a numpy array of size (n_samples, n_bridges) in which each row is a set of sampled bridge fragility parameters
	# U, a numpy array of size (num_gm_maps, n_bridges) in which each element is a sample from the standard uniform distribution
	# num_gm_maps, the number of earthquake scenarios to consider (of the 25 available)
	# num_damage_maps, the number of damage maps to create per scenario
	# returns f_X, a numpy array of size (n_samples,) in which each element is the output of mahmodel_road_only_napa.py

	n_samples = X.shape[0]

	print 'n_samples = ', n_samples

	# run the traffic model on each set of sampled fragility parameters and store the result of each run
	start = time.time()

	for i in range(0,n_samples):
		temp_damage_tracker = mahmodel.main_dam_only(i, map_indices, map_weights, bridge_ids,
																	   bridge_dict, X[i,:],
												   U, demand, damage_tracker, bridge_indices,
												   no_damage_travel_time, no_damage_vmt, no_damage_trips_made,
												   num_gm_maps, num_damage_maps)

		damage_tracker = temp_damage_tracker

	return damage_tracker

def compute_total_sobol_precomputed(f_X, f_V, normalize = True):
	# takes as input:
	# f_X, an array of size (n_samples,) in which each row is the function evaluated at a sample
	# f_V, an array of size (n_samples, n_dims) in which each element is the function evaluated at a hybrid point
	# normalize, a boolean indicating whether to return the normalized total-order Sobol indices
	# returns an array in which each element is the estimated total-order Sobol index for dimension i

	# to use with something other than the Ishigami function, implement your function of choice as a method and call it
	# in place of ishigami() below

	n_samples = f_X.shape[0]
	print 'f_V.shape ', f_V.shape
	n_dims = f_V.shape[1]

	print 'n_samples = ', n_samples, ' n_dims = ', n_dims

	tau = numpy.zeros((n_dims,)) # store the total Sobol indices in an array -- this is actually tau squared

	for i in range(0,n_dims): # i is u
		Sigma = 0

		for k in range(0,n_samples):
			# if i == 0:
			#     print f_X[k], f_V[k,i]
			#print f_X[k], f_V[k,i]
			Sigma += (f_X[k] - f_V[k, i])**2

		tau[i] = Sigma/(2*n_samples)

	if normalize: # normalize the total-order Sobol' indices
		var_hat = compute_sample_variance(f_X)
		print 'var_hat = ', var_hat, ' sum of f_X = ', sum(f_X)
		S = tau/var_hat
		return S
	else:
		return tau

def compute_first_order_sobol_precomputed_alt(f_X, f_Y, normalize=True): # TODO -- implements biased estimator per Owens' SIAM UQ 2016 notes
	# takes as input:
	# f_X, an array of size (n_samples,) in which each row is the function evaluated at a sample
	# f_Y, an array of size (n_samples, n_dims) in which each element is the function evaluated at a hybrid point
	# normalize, a boolean indicating whether to return the normalized total-order Sobol indices
	# returns an array in which each element is the estimated total-order Sobol index for dimension i

	n_samples = f_X.shape[0]
	n_dims = f_Y.shape[1]

	print 'n_samples = ', n_samples, ' n_dims = ', n_dims

	tau = numpy.zeros((n_dims,))

	for i in range(0, n_dims):  # i is u -- so the bridge (variable) of interest

		Sigma = 0

		# mu_hat = 0
		# for k in range(0, n_samples):
		# 	# print 'cbs: ', k, f_X[k], f_Y[k,i], (f_X[k] + f_Y[k,i])/2
		# 	mu_hat += f_X[k] + f_Y[k, i]

		mu_hat = sum(f_X)/ (n_samples)

		# print 'cbs mu_hat at i = ', i ,' = ', mu_hat, len(f_Y[:,i])

		for k in range(0, n_samples):
			Sigma += f_X[k] * f_Y[k, i] - mu_hat ** 2

		tau[i] = Sigma / (n_samples)

	if normalize:  # normalize the first-order Sobol' indices
		var_hat = compute_sample_variance(f_X)
		print 'var_hat = ', var_hat, ' sum of f_X = ', sum(f_X), 'mu_hat = ', mu_hat
		S = tau / var_hat
		return S
	else:
		return tau

def compute_first_order_sobol_precomputed(f_X, f_Y, normalize = True):
	# takes as input:
	# f_X, an array of size (n_samples,) in which each row is the function evaluated at a sample
	# f_Y, an array of size (n_samples, n_dims) in which each element is the function evaluated at a hybrid point
	# normalize, a boolean indicating whether to return the normalized total-order Sobol indices
	# returns an array in which each element is the estimated total-order Sobol index for dimension i

	n_samples = f_X.shape[0]
	n_dims = f_Y.shape[1]

	print 'n_samples = ', n_samples, ' n_dims = ', n_dims

	tau = numpy.zeros((n_dims,))
	mu_hats = []

	for i in range(0,n_dims): # i is u -- so the bridge (variable) of interest

		Sigma = 0

		mu_hat = 0
		for k in range(0,n_samples):
			#print 'cbs: ', k, f_X[k], f_Y[k,i], (f_X[k] + f_Y[k,i])/2
			mu_hat += f_X[k] + f_Y[k,i]

		mu_hat = mu_hat/(2*n_samples)
		mu_hats.append(mu_hat)

		# print 'cbs mu_hat at i = ', i ,' = ', mu_hat, len(f_Y[:,i])

		for k in range(0,n_samples):
			Sigma += f_X[k]*f_Y[k, i] - mu_hat**2

		tau[i] = Sigma/(n_samples)

	if normalize: # normalize the first-order Sobol' indices
		var_hat = compute_sample_variance(f_X)
		print 'var_hat = ', var_hat, ' sum of f_X = ', sum(f_X), ' average mu_hat across dimensions = ', numpy.average(mu_hats), min(mu_hats), max(mu_hats)
		S = tau/var_hat
		return S
	else:
		return tau

def get_sf_fullr_dict():

	with open('input/sf_fullr_dict.pkl', 'rb') as f:
		sf_dict = pickle.load(f)


	with open('input/sf_fullr_bridge_ids.pkl', 'rb') as f:
		bridge_ids = pickle.load(f)

	return sf_dict, bridge_ids

def run_sobol_computation(start_index, n_samples, partial_dict_name, scenarios, dam_maps_per_scenario, output_folder,
						  first_order = False, ret_cost = True):

	#u_start = time.time()

	directory = 'sobol_output/'
	folder = output_folder
	run_name = 'sf_fullr'
	filename = '_sf_fullr.pkl'

	# Declare input file names -- from binary distribution
	F_input = 'sobol_input/F_samples_'+run_name+'.pkl'  # N = 200, binary, custom omega
	F_prime_input = 'sobol_input/F_prime_samples_'+run_name+'.pkl'   # N = 200, custom omega
	# F_input = 'sobol_input/F_samples_sft2_new.pkl'  # N = 200
	# F_prime_input = 'sobol_input/F_prime_samples_sft2_new.pkl'  # N = 200
	#F_input = 'sobol_input/F_samples_sft2_new_2.pkl' # an additional N = 200 for sft2_new
	#F_prime_input = 'sobol_input/F_prime_samples_sft2_new_2.pkl' # an additional N = 200 for sft2_new
	# U_input = 'sobol_input/U_samples_sf_testbed_2_2.pkl' # for S = 20, D = 10; all bridges in sf_testbed_2 should get damaged at least once

	# declare output file names
	fX_times_output = directory + folder + 'fX_times' + filename # travel times for f_X
	fX_trips_output = directory + folder + 'fX_trips' + filename # trips made for f_X
	fX_vmts_output = directory + folder + 'fX_vmts' + filename # VMTs for f_X
	fX_avg_times_output = directory + folder + 'fX_avg_time' + filename  # average TT
	fX_avg_trips_output = directory + folder + 'fX_avg_trips' + filename # average trips made
	fX_avg_vmts_output = directory + folder + 'fX_avg_vmts' + filename # average VMT
	fX_delay_costs_output = directory + folder + 'fX_delay_costs' + filename
	fX_conn_costs_output = directory + folder + 'fX_conn_costs' + filename
	fX_indirect_costs_output = directory + folder + 'fX_indirect_costs' + filename
	fX_direct_costs_output = directory + folder + 'fX_direct_costs' + filename
	fX_exp_indirect_cost_output = directory + folder + 'fX_exp_indirect_costs' + filename
	fX_exp_direct_cost_output = directory + folder + 'fX_exp_direct_costs' + filename
	fX_expected_cost_output = directory + folder + 'fX_exp_costs' + filename
	fX_retrofit_cost_output = directory + folder + 'fX_ret_costs' + filename


	fV_times_output = directory + folder + 'fV_times' + filename  # travel times for f_X
	fV_trips_output = directory + folder + 'fV_trips' + filename  # trips made for f_X
	fV_vmts_output = directory + folder + 'fV_vmts' + filename  # VMTs for f_X
	fV_avg_times_output = directory + folder + 'fV_avg_time' + filename  # average TT
	fV_avg_trips_output = directory + folder + 'fV_avg_trips' + filename  # average trips made
	fV_avg_vmts_output = directory + folder + 'fV_avg_vmts' + filename  # average VMT
	fV_delay_costs_output = directory + folder + 'fV_delay_costs' + filename
	fV_conn_costs_output = directory + folder + 'fV_conn_costs' + filename
	fV_indirect_costs_output = directory + folder + 'fV_indirect_costs' + filename
	fV_direct_costs_output = directory + folder + 'fV_direct_costs' + filename
	fV_exp_indirect_cost_output = directory + folder + 'fV_exp_indirect_costs' + filename
	fV_exp_direct_cost_output = directory + folder + 'fV_exp_direct_costs' + filename
	fV_expected_cost_output = directory + folder + 'fV_exp_costs' + filename
	fV_retrofit_cost_output = directory + folder + 'fV_ret_costs' + filename

	damage_x_output = directory + folder + 'damage_x' +filename
	damage_v_output = directory + folder + 'damage_v' +filename
	S_cost_output = directory + folder + 'S_cost' + filename
	tau_cost_output = directory + folder + 'tau_cost' + filename

	sobol_index_dict_output = directory + folder + 'sobol_dict' + filename

	# load bridge dict, fragility function parameter samples, and uniform random number samples
	partial_dict, bridge_ids = get_sf_fullr_dict()

	print 'bridges = ', bridge_ids

	with open(F_input, 'rb') as f:
		F = pickle.load(f)

	print 'F shape = ', F.shape

	with open(F_prime_input, 'rb') as f:
		F_prime = pickle.load(f)

	print 'F_prime shape = ', F_prime.shape

	# with open('input/U_samples.pkl', 'rb') as f:
	#     U = pickle.load(f)

	n_bridges = len(bridge_ids)
	n_evals = (n_bridges+1)*(scenarios*dam_maps_per_scenario*n_samples)

	#if scenarios == 25:
	map_indices_input = 'sobol_input/' + run_name + '_training_map_indices.pkl'  # S = 30 for training sf_fullr
	map_weights_input = 'sobol_input/' + run_name + '_training_map_weights.pkl'  # S = 30 for training sf_fullr

	with open(map_indices_input,'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input,'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

	print len(map_indices), map_indices
	print len(map_weights), map_weights

	assert len(map_indices) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map indices list.'
	assert len(map_weights) == scenarios, 'Error: The number of scenarios (S) does not match the length of the map weights list.'

	print '****************************************************************************************'
	print 'Your settings will require ', n_evals, ' function evaluations (traffic model runs).'
	print 'N = ', n_samples # number of samples of fragility function parameters
	print 'S = ', scenarios # number of scenarios to consider -- equivalent to ground-motion maps since we are only considering 1 GM map per scenario
	print 'D = ', dam_maps_per_scenario
	print 'B = ', n_bridges
	print 'first-order? ', first_order
	print '****************************************************************************************'


	with open('sobol_input/U_good_sf_fullr.pkl', 'rb') as f:
		U_temp = pickle.load(f)

	if scenarios < 1992:  # get subset of uniform random numbers that correspond to the scenarios of interest
		U = numpy.zeros((scenarios * dam_maps_per_scenario, n_bridges))
		i = 0
		for s in map_indices:
			U[i * dam_maps_per_scenario:(i + 1) * dam_maps_per_scenario, :] = U_temp[s * dam_maps_per_scenario:(s + 1) * dam_maps_per_scenario,
																  :]
			i += 1

	assert U.shape[0] == scenarios * dam_maps_per_scenario, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
	assert U.shape[1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'

	print 'U shapes (set and subset): ', U_temp.shape, U.shape

	# Get samples of the fragility function parameters and create hybrid points.

	small_F = F[start_index:start_index+n_samples,]
	small_F_prime = F_prime[start_index:start_index+n_samples,]
	small_V = interleave_set(small_F, small_F_prime,first_order=first_order)

	print 'small F shape: ', small_F.shape
	print 'small V shape: ', small_V.shape

	# Set up traffic model and run it.
	G = mahmodel.get_graph()

	assert G.is_multigraph() == False, 'You want a directed graph without multiple edges between nodes'

	demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
							 'input/superdistricts_centroids_dummies.csv')

	no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = precompute_network_performance(G, demand)


	print 'no_damage_travel_time = ', no_damage_travel_time
	print 'no_damage_vmt = ', no_damage_vmt
	print 'no_damage_trips_made = ', no_damage_trips_made

	print 'starting f_X'

	# # Keep track of which bridges get damaged when computing f_X.
	damage_tracker = numpy.zeros((scenarios*dam_maps_per_scenario,n_bridges,n_samples)) # array of size (SxD, B, N)
	bridge_indices = {bridge_ids[i]:i for i in range(0,len(bridge_ids))} # each bridge has an index in the damage_tracker array


	f_X_times, f_X_trips, f_X_vmts, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, f_X_direct_costs, \
	f_X_avg_time, f_X_avg_trip, f_X_avg_vmt, f_X_exp_indirect_cost, f_X_exp_direct_cost, f_X_exp_cost, f_X_retrofit_cost, damage_tracker_x = run_traffic_model_set(bridge_ids, partial_dict, map_indices, map_weights, small_F, U, demand,
												  damage_tracker, bridge_indices, no_damage_travel_time,
												  no_damage_vmt, no_damage_trips_made, num_gm_maps= scenarios,
												  num_damage_maps=dam_maps_per_scenario, ret_cost=ret_cost)

	print f_X_avg_time, f_X_avg_trip, f_X_exp_indirect_cost

	# save data for f_X
	with open(damage_x_output,'wb') as f:
		pickle.dump(damage_tracker_x, f)

	with open(fX_times_output, 'wb') as f: # save raw performance data
		pickle.dump(f_X_times, f)
	with open(fX_trips_output, 'wb') as f:
		pickle.dump(f_X_trips, f)
	with open(fX_vmts_output, 'wb') as f:
		pickle.dump(f_X_vmts, f)

	with open(fX_avg_times_output, 'wb') as f: # save average (expected) performance data
		pickle.dump(f_X_avg_time, f)
	with open(fX_avg_trips_output, 'wb') as f:
		pickle.dump(f_X_avg_trip, f)
	with open(fX_avg_vmts_output, 'wb') as f:
		pickle.dump(f_X_avg_vmt, f)

	with open(fX_delay_costs_output, 'wb') as f:
		pickle.dump(f_X_delay_costs, f)
	with open(fX_conn_costs_output, 'wb') as f:
		pickle.dump(f_X_conn_costs, f)
	with open(fX_direct_costs_output, 'wb') as f:
		pickle.dump(f_X_direct_costs, f)
	with open(fX_indirect_costs_output, 'wb') as f:
		pickle.dump(f_X_indirect_costs, f)

	with open(fX_exp_direct_cost_output, 'wb') as f:
		pickle.dump(f_X_exp_direct_cost, f)
	with open(fX_exp_indirect_cost_output, 'wb') as f:
		pickle.dump(f_X_exp_indirect_cost, f)
	with open(fX_expected_cost_output, 'wb') as f:
		pickle.dump(f_X_exp_cost, f)
	with open(fX_retrofit_cost_output, 'wb') as f:
		pickle.dump(f_X_retrofit_cost, f)


	# keep track of the damage that occurs when computing f_V
	damage_tracker_v = numpy.zeros((scenarios*dam_maps_per_scenario,n_bridges,n_samples,n_bridges)) # array of size (SxD, B, N, B)

	f_V_times = numpy.ones((n_samples, scenarios*dam_maps_per_scenario, n_bridges))
	f_V_trips = numpy.ones((n_samples, scenarios*dam_maps_per_scenario, n_bridges))
	f_V_vmts = numpy.ones((n_samples, scenarios*dam_maps_per_scenario, n_bridges))

	f_V_avg_time = numpy.ones((n_samples, n_bridges))
	f_V_avg_trip = numpy.ones((n_samples, n_bridges))
	f_V_avg_vmt = numpy.ones((n_samples, n_bridges))

	f_V_delay_costs = numpy.ones((n_samples, scenarios*dam_maps_per_scenario, n_bridges))
	f_V_conn_costs = numpy.ones((n_samples, scenarios*dam_maps_per_scenario, n_bridges))
	f_V_indirect_costs = numpy.ones((n_samples, scenarios*dam_maps_per_scenario, n_bridges))
	f_V_direct_costs = numpy.ones((n_samples, scenarios*dam_maps_per_scenario, n_bridges))
	f_V_exp_indirect_cost = numpy.ones((n_samples, n_bridges))
	f_V_exp_direct_cost = numpy.ones((n_samples, n_bridges))
	f_V_exp_cost = numpy.ones((n_samples, n_bridges))
	f_V_ret_cost = numpy.ones((n_samples, n_bridges))


	print 'starting f_V computation'
	f_V_start = time.time()

	for i in range(0, 1):#n_bridges): # TODO PUT BACK TO N_BRIDGES

		temp_times, temp_trips, temp_vmts, temp_delay_costs, temp_conn_costs, temp_indirect_costs, temp_direct_costs, temp_avg_time, \
		temp_avg_trip, temp_avg_vmt, temp_exp_indirect_cost, temp_exp_direct_cost, temp_exp_cost, temp_retrofit_cost, temp_damage_tracker_v = run_traffic_model_set(bridge_ids,partial_dict, map_indices, map_weights,
																 small_V[:,:,i], U, demand, damage_tracker_v[:,:,:,i],
																 bridge_indices, no_damage_travel_time, no_damage_vmt, no_damage_trips_made,
																 num_gm_maps= scenarios,
																 num_damage_maps=dam_maps_per_scenario, ret_cost=ret_cost)


		f_V_times[:, :, i] = temp_times
		f_V_trips[:, :, i] = temp_trips
		f_V_vmts[:, :, i] = temp_vmts

		f_V_delay_costs[:, :, i] = temp_delay_costs
		f_V_conn_costs[:, :, i] = temp_conn_costs
		f_V_indirect_costs[:, :, i] = temp_indirect_costs
		f_V_direct_costs[:, :, i] = temp_direct_costs

		f_V_avg_time[:, i] = temp_avg_time
		f_V_avg_trip[:, i] = temp_avg_trip
		f_V_avg_vmt[:, i] = temp_avg_vmt

		f_V_exp_indirect_cost[:, i] = temp_exp_indirect_cost
		f_V_exp_direct_cost[:, i] = temp_exp_direct_cost
		f_V_exp_cost[:, i] = temp_exp_cost
		f_V_ret_cost[:, i] = temp_retrofit_cost


		damage_tracker_v[:, :, :, i] = temp_damage_tracker_v

		print temp_times
		print 'computing f_V, done with i = ', i, ' of ', n_bridges - 1, ' took ', time.time()-f_V_start, ' seconds' # TODO -- MAKE SURE UPPER BOUND IS N_SAMPLES

	# save data for f_V
	with open(damage_v_output, 'wb') as f:
		pickle.dump(damage_tracker_v, f)

	with open(fV_times_output, 'wb') as f:  # save raw performance data
		pickle.dump(f_V_times, f)
	with open(fV_trips_output, 'wb') as f:
		pickle.dump(f_V_trips, f)
	with open(fV_vmts_output, 'wb') as f:
		pickle.dump(f_V_vmts, f)

	with open(fV_avg_times_output, 'wb') as f:  # save average (expected) performance data
		pickle.dump(f_V_avg_time, f)
	with open(fV_avg_trips_output, 'wb') as f:
		pickle.dump(f_V_avg_trip, f)
	with open(fV_avg_vmts_output, 'wb') as f:
		pickle.dump(f_V_avg_vmt, f)

	with open(fV_delay_costs_output, 'wb') as f:
		pickle.dump(f_V_delay_costs, f)
	with open(fV_conn_costs_output, 'wb') as f:
		pickle.dump(f_V_conn_costs, f)
	with open(fV_indirect_costs_output, 'wb') as f:
		pickle.dump(f_V_indirect_costs, f)
	with open(fV_direct_costs_output, 'wb') as f:
		pickle.dump(f_V_direct_costs, f)

	with open(fV_exp_direct_cost_output, 'wb') as f:
		pickle.dump(f_V_exp_direct_cost, f)
	with open(fV_exp_indirect_cost_output, 'wb') as f:
		pickle.dump(f_V_exp_indirect_cost, f)
	with open(fV_expected_cost_output, 'wb') as f:
		pickle.dump(f_V_exp_cost, f)
	with open(fV_retrofit_cost_output, 'wb') as f:
		pickle.dump(f_V_ret_cost, f)

	with open(damage_v_output, 'wb') as f:  # save latest version of damage_tracker_v array
		pickle.dump(damage_tracker_v, f)

	print 'done with all f_V computation -- took ', time.time()-f_V_start, ' seconds'

	# Compute Sobol indices.
	s_start = time.time()

	if first_order:
		print '*** FIRST-ORDER INDEX RESULTS ***'
		S_cost = compute_first_order_sobol_precomputed(f_X_exp_cost,f_V_exp_cost)

		print 'S_cost = ', S_cost
		print 'computing S took ', time.time() - s_start, ' seconds'

		with open(S_cost_output, 'wb') as f:
			pickle.dump(S_cost, f)

		tau_start = time.time()
		tau_cost = compute_first_order_sobol_precomputed(f_X_exp_cost, f_V_exp_cost, normalize=False)

		print 'tau_cost = ', tau_cost

		print 'computing tau took ', time.time() - tau_start, ' seconds'

		with open(tau_cost_output, 'wb') as f:
			pickle.dump(tau_cost, f)

		sobol_dict = {}

		for i in range(0, n_bridges):
			sobol_dict[str(bridge_ids[i])] = {}
			sobol_dict[str(bridge_ids[i])]['S_cost'] = S_cost[i]

		with open(sobol_index_dict_output, 'wb') as f:
			pickle.dump(sobol_dict, f)

	else:
		print '*** TOTAL-ORDER INDEX RESULTS ***'
		S_cost = compute_total_sobol_precomputed(f_X_exp_cost,f_V_exp_cost)


		print 'S_cost = ', S_cost

		print 'computing S took ', time.time() - s_start, ' seconds'

		with open(S_cost_output, 'wb') as f:
			pickle.dump(S_cost,f)

		tau_start = time.time()
		tau_cost = compute_total_sobol_precomputed(f_X_exp_cost, f_V_exp_cost, normalize=False)

		print 'tau_cost = ', tau_cost

		print 'computing tau took ', time.time() - tau_start, ' seconds'

		with open(tau_cost_output, 'wb') as f:
			pickle.dump(tau_cost,f)

		sobol_dict = {}

		for i in range(0,n_bridges):
			sobol_dict[str(bridge_ids[i])] = {}
			sobol_dict[str(bridge_ids[i])]['S_cost'] = S_cost[i]

		with open(sobol_index_dict_output,'wb') as f:
			pickle.dump(sobol_dict,f)

	for b in bridge_ids:
		print b, sobol_dict[b]['S_cost']

	print 'Congratulations! Your run has been completed successfully.'

	# print '*** Times'
	# print f_X_times
	# print f_X_avg_time
	# print f_V_times
	# print f_V_avg_time
	#
	# print '*** Costs'
	# print f_X_delay_costs
	# print f_X_conn_costs
	# print f_V_delay_costs

# compare_bridge_damage_rates(10, scenarios=30)
# get_bridge_damage_rates(1, False, 10) # in cs1, bridge 49 does not get damaged when unretrofitted; in cs2, same with bridge 1475

# # Find a single set of uniform random numbers that result in non-zero rates of bridge damage for unretrofitted bridges in CS1, CS2, and CS3.
# # Then save it for use with all 3 case studies.
# sc = False
# while sc == False:
# 	iterate_damage_rates(1, False, scenarios=25) # find a U that works for cs 1
# 	sc2 = get_bridge_damage_rates(2, False, 10) # test cs2 using U that works for cs1
# 	sc3 = get_bridge_damage_rates(3, False, 10) # test cs3 using U that works for cs1
# 	if sc2 == True and sc3 == True:
# 		sc = True
#
# with open('sobol_output/cs1/U_good.pkl', 'rb') as f:
# 	U_good = pickle.load(f)
#
# with open('sobol_input/U_cs_s1992d10.pkl', 'wb') as f:
# 	pickle.dump(U_good, f)


# generate a good U for cs9
# iterate_damage_rates(9, p = False, scenarios=20)

# check whether U that works for cs1, cs2, cs3 also works for cs9 -- IT DOES NOT. Bridge 941 never gets damaged.
# get_bridge_damage_rates(9, p = False, dam_maps_per_scenario=10, scenarios=20)

# get_bridge_damage_rates(cs_number=3, p = False, dam_maps_per_scenario= 10, scenarios = 27, test = True)

# STUFF FOR sf_fullr
# iterate_damage_rates_sf_fullr(p=False,scenarios=30,dam_maps_per_scenario=10)
# sc = get_bridge_damage_rates_sf_fullr(p=False,dam_maps_per_scenario=10,scenarios=30,test=False)