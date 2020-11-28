from __future__ import division

import pickle, csv, util
import numpy as np
from compute_bridge_sobol_sf_full import precompute_network_performance, compute_sample_variance
from process_results_sf import load_individual_undamaged_stats
import mahmodel_road_only as mahmodel
import bd_test as bd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import bridges_plot.bridges as bplot
import plotly
import plotly.graph_objs as go
from math import log

alpha = 48 # dollars per hour
beta = 78*8 # dollars per hour times hours

# GRAPHICS METHODS
def save_color_dict():

	color_dict = {}
	color_dict['age'] = '#0072B2'
	color_dict['oldest'] = '#0072B2'
	color_dict['fragility'] = '#56B4E9'
	color_dict['weakest'] = '#56B4E9'
	color_dict['traffic'] = '#E69F00'
	color_dict['busiest'] = '#E69F00'
	color_dict['composite'] = '#009E73'
	color_dict['OAT'] = '#D55E00'
	color_dict['OAT total'] = '#D55E00'
	color_dict['Sobol, exp. cost'] = '#CC79A7'
	color_dict['Sobol exp. total'] = '#CC79A7'
	color_dict['Sobol, perc'] = '#CC79A7'
	color_dict['Sobol'] = '#CC79A7'
	color_dict['p0.2'] = '#DDACE3'
	color_dict['p0.8'] = '#824E49'

	with open('color_dict.pkl','wb') as f:
		pickle.dump(color_dict,f)

def get_color_dict():

	with open('color_dict.pkl','rb') as f:
		color_dict = pickle.load(f)

	return color_dict

def get_color(series):

	color_dict = get_color_dict()

	return color_dict[series]
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def compute_weighted_average_performance(lnsas, map_weights, num_damage_maps, travel_times, vmts, trips_made,
										 no_damage_travel_time, no_damage_vmt, no_damage_trips_made, direct_costs):
	# Compute weighted average of performance metrics for a single sample of a fragility function vector.

	scenarios = len(lnsas)  # number of scenarios under consideration

	# GB ADDITION -- computed weighted average (expectation) of travel time and other metrics of interest
	average_travel_time = 0
	average_trips_made = 0
	average_vmt = 0
	average_direct_costs = 0

	for j in range(0, len(lnsas)):  # for every scenario considered
		w = map_weights[j]
		temp_times = np.asarray(travel_times[np.arange(start=j, stop=scenarios * num_damage_maps, step=scenarios)])
		temp_trips = np.asarray(trips_made[np.arange(start=j, stop=scenarios * num_damage_maps, step=scenarios)])
		temp_vmts = np.asarray(vmts[np.arange(start=j, stop=scenarios * num_damage_maps, step=scenarios)])
		temp_direct_costs = np.asarray(
			direct_costs[np.arange(start=j, stop=scenarios * num_damage_maps, step=scenarios)])
		temp_time_cost = [alpha*(t - no_damage_travel_time) for t in temp_times]
		temp_conn_cost = [beta*(no_damage_trips_made-t) for t in trips_made]
		temp_cost = [temp_time_cost[i] + temp_conn_cost[i] for i in range(0,len(temp_time_cost))]
		# print('j = ', j, np.var(temp_cost))
		assert temp_trips.shape[0] == num_damage_maps, 'Error -- wrong number of trips.'
		assert temp_times.shape[0] == num_damage_maps, 'Error -- wrong number of times.'
		assert temp_vmts.shape[0] == num_damage_maps, 'Error -- wrong number of vmts.'
		average_travel_time += w * np.average(temp_times)
		average_trips_made += w * np.average(temp_trips)
		average_vmt += w * np.average(temp_vmts)
		average_direct_costs += w * np.average(temp_direct_costs)

	# add the scenario of no earthquake
	average_travel_time += (1 - sum(map_weights)) * no_damage_travel_time
	average_trips_made += (1 - sum(map_weights)) * no_damage_trips_made
	average_vmt += (1 - sum(map_weights)) * no_damage_vmt

	average_delay_cost = alpha * max(0, ((
													 average_travel_time - no_damage_travel_time) / 3600))  # travel times are in seconds, so convert to units of monetary units/hour*hours --> monetary units per day; assume travel times increase with damage

	average_connectivity_cost = beta * max(0, (
				no_damage_trips_made - average_trips_made))  # units of monetary units/hour*lost trips/day*hours/(trips*days)--> monetary units per day; assume total flows decrease with damage

	assert average_delay_cost >= 0, 'ERROR in compute_indirect_costs(): delay cost is negative.'
	assert average_connectivity_cost >= 0, 'ERROR in compute_indirect_costs(): connectivity cost is negative.'

	average_indirect_cost = average_delay_cost + average_connectivity_cost

	return average_travel_time, average_vmt, average_trips_made, average_direct_costs, average_delay_cost, average_connectivity_cost, average_indirect_cost
def compute_percent_reduction(baseline, new_value):

	return (new_value-baseline)/baseline*100

def get_retrofit_results_from_file(n_retrofits, strategy, no_damage_travel_time, no_damage_vmt, no_damage_trips_made, training): #TODO -- for revised average retrofit results

	# strategies = ['oldest', 'busiest', 'weakest', 'composite', 'OAT total', 'OAT indirect', 'OAT direct', 'Sobol exp. total', 'Sobol exp. indirect', 'Sobol exp. direct'] # TODO -- ORIGINAL
	strategies = ['oldest', 'busiest', 'weakest', 'composite', 'OAT total', 'Sobol exp. total'] # TODO -- for revised average retrofit results

	strategy_index = strategies.index(strategy)

	# output_folder = 'sobol_output/retrofits/max_cost/max_cost_' + str(n_retrofits) + '/'
	if training: # S = 30
		scenarios = 30
		# output_folder = 'sobol_output/retrofits/s30/r' + str(n_retrofits) + '/'
		output_folder = 'sobol_output/retrofits/ret_revised_avg_s30/r' + str(n_retrofits) + '/'
		map_indices_input = 'sobol_input/sf_fullr_training_map_indices.pkl'  # S = 30 for training sf_fullr
		map_weights_input = 'sobol_input/sf_fullr_training_map_weights.pkl'  # S = 30 for training sf_fullr
	else: # S = 45
		scenarios = 45
		# output_folder = 'sobol_output/retrofits/rets_to_local/r' + str(n_retrofits) + '/'
		output_folder = 'sobol_output/retrofits/ret_revised_avg/r' + str(n_retrofits) + '/'
		if strategy == 'Sobol exp. total':
			output_folder = 'sobol_output/retrofits/sobol_retrofits/r' + str(n_retrofits) + '/'
		map_indices_input = 'sobol_input/sf_fullr_testing_map_indices.pkl'  # S = 45 for testing sf_fullr
		map_weights_input = 'sobol_input/sf_fullr_testing_map_weights.pkl'  # S = 45 for testing sf_fullr

	filename = '_sf_fullr'

	# store the results
	fX_times_output = output_folder + 'fX_times' + filename  # travel times for f_X
	fX_trips_output = output_folder + 'fX_trips' + filename  # trips made for f_X
	fX_vmts_output = output_folder + 'fX_vmts' + filename  # VMTs for f_X
	fX_avg_times_output = output_folder + 'fX_avg_time' + filename  # average TT
	fX_avg_trips_output = output_folder + 'fX_avg_trips' + filename  # average trips made
	fX_avg_vmts_output = output_folder + 'fX_avg_vmts' + filename  # average VMT
	fX_delay_costs_output = output_folder + 'fX_delay_costs' + filename
	fX_conn_costs_output = output_folder + 'fX_conn_costs' + filename
	fX_indirect_costs_output = output_folder + 'fX_indirect_costs' + filename
	fX_direct_costs_output = output_folder + 'fX_direct_costs' + filename
	fX_exp_indirect_cost_output = output_folder + 'fX_exp_indirect_costs' + filename
	fX_exp_direct_cost_output = output_folder + 'fX_exp_direct_costs' + filename
	fX_expected_cost_output = output_folder + 'fX_exp_costs' + filename

	damage_x_output = output_folder + 'damage_x' + filename

	# save data for f_X
	with open(damage_x_output, 'rb') as f:
		damage_tracker = pickle.load(f)

	with open(fX_times_output, 'rb') as f:  # save raw performance data
		f_X_times = pickle.load(f)
	with open(fX_trips_output, 'rb') as f:
		f_X_trips = pickle.load(f)
	with open(fX_vmts_output, 'rb') as f:
		f_X_vmts = pickle.load(f)

	with open(fX_avg_times_output, 'rb') as f:  # save average (expected) performance data
		f_X_avg_time = pickle.load(f)
	with open(fX_avg_trips_output, 'rb') as f:
		f_X_avg_trip = pickle.load(f)
	with open(fX_avg_vmts_output, 'rb') as f:
		f_X_avg_vmt = pickle.load(f)

	with open(fX_delay_costs_output, 'rb') as f:
		f_X_delay_costs = pickle.load(f)
	with open(fX_conn_costs_output, 'rb') as f:
		f_X_conn_costs = pickle.load(f)
	with open(fX_direct_costs_output, 'rb') as f:
		f_X_direct_costs = pickle.load(f)
	with open(fX_indirect_costs_output, 'rb') as f:
		f_X_indirect_costs = pickle.load(f)

	with open(fX_exp_direct_cost_output, 'rb') as f:
		f_X_exp_direct_cost = pickle.load(f)
	with open(fX_exp_indirect_cost_output, 'rb') as f:
		f_X_exp_indirect_cost = pickle.load(f)
	with open(fX_expected_cost_output, 'rb') as f:
		f_X_exp_cost = pickle.load(f)

	# n_samples = f_X_delay_costs.shape[0]
	# print 'shapes = ', f_X_indirect_costs.shape

	# print f_X_delay_costs.shape, f_X_delay_costs[strategy_index]


	with open(map_indices_input,'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input,'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

	## GB: this gets hazard-consistent maps that we created from Miller's subsetting procedure
	sa_matrix_full = util.read_2dlist('input/sample_ground_motion_intensity_maps_road_only_filtered.txt',
									  delimiter='\t')
	sa_matrix = [sa_matrix_full[i] for i in map_indices]  # GB: get the ground_motions for just the scenarios we are interested in


	lnsas = []
	magnitudes = []
	for row in sa_matrix:
		lnsas.append([log(float(sa)) for sa in row[4:]])
		magnitudes.append(float(row[2]))

	# get just the information relating to the strategy of interest
	if n_retrofits > 0 and n_retrofits < 71:
		# print strategy, strategy_index, n_retrofits, f_X_times.shape, f_X_times[strategy_index].shape

		average_travel_time, average_vmt, average_trips_made, average_direct_cost, average_delay_cost, average_connectivity_cost, \
		average_indirect_cost = compute_weighted_average_performance(lnsas, map_weights, num_damage_maps=10,
																	 travel_times=f_X_times[strategy_index],
																	 vmts=f_X_vmts[strategy_index],
																	 trips_made=f_X_trips[strategy_index],
																	 no_damage_travel_time=no_damage_travel_time,
																	 no_damage_vmt=no_damage_vmt,
																	 no_damage_trips_made=no_damage_trips_made,
																	 direct_costs=f_X_direct_costs[strategy_index])

		f_X_avg_time = average_travel_time
		f_X_avg_vmt = average_vmt
		f_X_avg_trip = average_trips_made
		f_X_exp_direct_cost = average_direct_cost
		f_X_exp_indirect_cost = average_indirect_cost  # hourly
		f_X_exp_cost = 24 * 125 * average_indirect_cost + average_direct_cost
		# f_X_delay_costs = average_delay_cost
		# f_X_conn_costs = average_connectivity_cost

		# f_X_delay_costs = f_X_delay_costs[strategy_index]
		# f_X_conn_costs = f_X_conn_costs[strategy_index]
		# f_X_direct_costs = f_X_direct_costs[strategy_index]
		# f_X_indirect_costs = f_X_indirect_costs[strategy_index]
		# f_X_exp_cost = f_X_exp_cost[strategy_index]


	# return f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, f_X_direct_costs, f_X_exp_cost
	return f_X_exp_cost

def get_baseline_retrofit_results(n_retrofits, no_damage_travel_time, no_damage_vmt, no_damage_trips_made, training, detailed=False): # assuming S = 45 #TODO -- this is for retrofits with revised average

	filename = '_sf_fullr'

	if not training:
		output_folder = 'sobol_output/retrofits/ret_revised_avg/r' + str(n_retrofits) + '/'
		scenarios = 45
		map_indices_input = 'sobol_input/sf_fullr_testing_map_indices.pkl'  # S = 45 for training sf_fullr
		map_weights_input = 'sobol_input/sf_fullr_testing_map_weights.pkl'  # S = 45 for training sf_fullr
	else:
		output_folder = 'sobol_output/retrofits/ret_revised_avg_s30/r' + str(n_retrofits) + '/'
		scenarios = 30
		map_indices_input = 'sobol_input/sf_fullr_training_map_indices.pkl'  # S = 30 for training sf_fullr
		map_weights_input = 'sobol_input/sf_fullr_training_map_weights.pkl'  # S = 30 for training sf_fullr

	# store the results
	fX_times_output = output_folder + 'fX_times' + filename  # travel times for f_X
	fX_trips_output = output_folder + 'fX_trips' + filename  # trips made for f_X
	fX_vmts_output = output_folder + 'fX_vmts' + filename  # VMTs for f_X
	fX_avg_times_output = output_folder + 'fX_avg_time' + filename  # average TT
	fX_avg_trips_output = output_folder + 'fX_avg_trips' + filename  # average trips made
	fX_avg_vmts_output = output_folder + 'fX_avg_vmts' + filename  # average VMT
	fX_delay_costs_output = output_folder + 'fX_delay_costs' + filename
	fX_conn_costs_output = output_folder + 'fX_conn_costs' + filename
	fX_indirect_costs_output = output_folder + 'fX_indirect_costs' + filename
	fX_direct_costs_output = output_folder + 'fX_direct_costs' + filename
	fX_exp_indirect_cost_output = output_folder + 'fX_exp_indirect_costs' + filename
	fX_exp_direct_cost_output = output_folder + 'fX_exp_direct_costs' + filename
	fX_expected_cost_output = output_folder + 'fX_exp_costs' + filename

	damage_x_output = output_folder + 'damage_x' + filename

	# save data for f_X
	with open(damage_x_output, 'rb') as f:
		damage_tracker = pickle.load(f)

	with open(fX_times_output, 'rb') as f:  # save raw performance data
		f_X_times = pickle.load(f)
	with open(fX_trips_output, 'rb') as f:
		f_X_trips = pickle.load(f)
	with open(fX_vmts_output, 'rb') as f:
		f_X_vmts = pickle.load(f)

	with open(fX_avg_times_output, 'rb') as f:  # save average (expected) performance data
		f_X_avg_time = pickle.load(f)
	with open(fX_avg_trips_output, 'rb') as f:
		f_X_avg_trip = pickle.load(f)
	with open(fX_avg_vmts_output, 'rb') as f:
		f_X_avg_vmt = pickle.load(f)

	with open(fX_delay_costs_output, 'rb') as f:
		f_X_delay_costs = pickle.load(f)
	with open(fX_conn_costs_output, 'rb') as f:
		f_X_conn_costs = pickle.load(f)
	with open(fX_direct_costs_output, 'rb') as f:
		f_X_direct_costs = pickle.load(f)
	with open(fX_indirect_costs_output, 'rb') as f:
		f_X_indirect_costs = pickle.load(f)

	with open(fX_exp_direct_cost_output, 'rb') as f:
		f_X_exp_direct_cost = pickle.load(f)
	with open(fX_exp_indirect_cost_output, 'rb') as f:
		f_X_exp_indirect_cost = pickle.load(f)
	with open(fX_expected_cost_output, 'rb') as f:
		f_X_exp_cost = pickle.load(f)



	with open(map_indices_input, 'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input, 'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

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

	# no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = precompute_network_performance()

	# get just the information relating to the strategy of interest
	if n_retrofits == 0 or n_retrofits == 71:
		# print strategy, strategy_index, n_retrofits, f_X_times.shape, f_X_times[strategy_index].shape

		average_travel_time, average_vmt, average_trips_made, average_direct_cost, average_delay_cost, average_connectivity_cost, \
		average_indirect_cost = compute_weighted_average_performance(lnsas, map_weights, num_damage_maps=10,
																	 travel_times=f_X_times[0],
																	 vmts=f_X_vmts[0],
																	 trips_made=f_X_trips[0],
																	 no_damage_travel_time=no_damage_travel_time,
																	 no_damage_vmt=no_damage_vmt,
																	 no_damage_trips_made=no_damage_trips_made,
																	 direct_costs=f_X_direct_costs[0])

		f_X_avg_time = average_travel_time
		f_X_avg_vmt = average_vmt
		f_X_avg_trip = average_trips_made
		f_X_exp_direct_cost = average_direct_cost
		f_X_exp_indirect_cost = average_indirect_cost  # hourly
		f_X_exp_cost = 24 * 125 * average_indirect_cost + average_direct_cost

	if not detailed:
		return f_X_exp_cost
	else:
		return f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, f_X_direct_costs, f_X_exp_cost

def get_retrofit_results(output_folder, n_scenarios, filename='_sf_full', print_results=True): #TODO -- correct expectation computation

	scenarios = n_scenarios

	# store the results
	fX_times_output = output_folder + 'fX_times' + filename  # travel times for f_X
	fX_trips_output = output_folder + 'fX_trips' + filename  # trips made for f_X
	fX_vmts_output = output_folder + 'fX_vmts' + filename  # VMTs for f_X
	fX_avg_times_output = output_folder + 'fX_avg_time' + filename  # average TT
	fX_avg_trips_output = output_folder + 'fX_avg_trips' + filename  # average trips made
	fX_avg_vmts_output = output_folder + 'fX_avg_vmts' + filename  # average VMT
	fX_delay_costs_output = output_folder + 'fX_delay_costs' + filename
	fX_conn_costs_output = output_folder + 'fX_conn_costs' + filename
	fX_indirect_costs_output = output_folder + 'fX_indirect_costs' + filename
	fX_direct_costs_output = output_folder + 'fX_direct_costs' + filename
	fX_exp_indirect_cost_output = output_folder + 'fX_exp_indirect_costs' + filename
	fX_exp_direct_cost_output = output_folder + 'fX_exp_direct_costs' + filename
	fX_expected_cost_output = output_folder + 'fX_exp_costs' + filename
	#
	# damage_x_output = output_folder + 'damage_x' + filename
	#
	# # save data for f_X
	# with open(damage_x_output, 'rb') as f:
	# 	damage_tracker = pickle.load(f)

	with open(fX_times_output, 'rb') as f:  # save raw performance data
		f_X_times = pickle.load(f)
	with open(fX_trips_output, 'rb') as f:
		f_X_trips = pickle.load(f)
	with open(fX_vmts_output, 'rb') as f:
		f_X_vmts = pickle.load(f)

	with open(fX_avg_times_output, 'rb') as f:  # save average (expected) performance data
		f_X_avg_time = pickle.load(f)
	with open(fX_avg_trips_output, 'rb') as f:
		f_X_avg_trip = pickle.load(f)
	with open(fX_avg_vmts_output, 'rb') as f:
		f_X_avg_vmt = pickle.load(f)

	with open(fX_delay_costs_output, 'rb') as f:
		f_X_delay_costs = pickle.load(f)
	with open(fX_conn_costs_output, 'rb') as f:
		f_X_conn_costs = pickle.load(f)
	with open(fX_direct_costs_output, 'rb') as f:
		f_X_direct_costs = pickle.load(f)
	with open(fX_indirect_costs_output, 'rb') as f:
		f_X_indirect_costs = pickle.load(f)

	with open(fX_exp_direct_cost_output, 'rb') as f:
		f_X_exp_direct_cost = pickle.load(f)
	with open(fX_exp_indirect_cost_output, 'rb') as f:
		f_X_exp_indirect_cost = pickle.load(f)
	with open(fX_expected_cost_output, 'rb') as f:
		f_X_exp_cost = pickle.load(f)

	# print 'f_X_times.shape', f_X_times.shape
	batch_size = f_X_times.shape[0]

	# Get the weighted average of all metrics of interest using the updated calculation and raw results.
	tt0, vmt0, trips0 = load_individual_undamaged_stats()

	if scenarios == 30:
		map_indices_input = 'sobol_input/sf_fullr_training_map_indices.pkl'  # S = 30 for training sf_fullr
		map_weights_input = 'sobol_input/sf_fullr_training_map_weights.pkl'  # S = 30 for training sf_fullr
	elif scenarios == 45:
		map_indices_input = 'sobol_input/sf_fullr_testing_map_indices.pkl'  # S = 30 for training sf_fullr
		map_weights_input = 'sobol_input/sf_fullr_testing_map_weights.pkl'  # S = 30 for training sf_fullr
	else:
		print 'Need 30 or 45 scenarios.'

	with open(map_indices_input, 'rb') as f:
		map_indices = pickle.load(f)

	with open(map_weights_input, 'rb') as f:
		map_weights = pickle.load(f)

	if len(map_indices) != scenarios:
		map_indices = map_indices[0]
		map_weights = map_weights[0]

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

	temp_fX_avg_times = np.zeros((batch_size,))
	temp_fX_avg_vmts = np.zeros((batch_size,))
	temp_fX_avg_trips = np.zeros((batch_size,))
	temp_fX_exp_indirect_cost = np.zeros((batch_size,))
	temp_fX_exp_direct_cost = np.zeros((batch_size,))
	temp_fX_expected_cost = np.zeros((batch_size,))

	for k in range(0, batch_size):
		# print '*** batch = ', i, ' sample = ', k
		average_travel_time, average_vmt, average_trips_made, average_direct_cost, average_delay_cost, average_connectivity_cost, \
		average_indirect_cost = compute_weighted_average_performance(lnsas, map_weights, num_damage_maps=10,
																	 travel_times=f_X_times[k, :],
																	 vmts=f_X_vmts[k, :],
																	 trips_made=f_X_trips[k, :],
																	 no_damage_travel_time=tt0,
																	 no_damage_vmt=vmt0,
																	 no_damage_trips_made=trips0,
																	 direct_costs=f_X_direct_costs[k, :])

		temp_fX_avg_times[k] = average_travel_time
		temp_fX_avg_vmts[k] = average_vmt
		temp_fX_avg_trips[k] = average_trips_made
		temp_fX_exp_direct_cost[k] = average_direct_cost
		temp_fX_exp_indirect_cost[k] = average_indirect_cost  # hourly
		temp_fX_expected_cost[k] = 24 * 125 * average_indirect_cost + average_direct_cost

	assert np.any(temp_fX_exp_indirect_cost == 0) == False, 'Error in correcting fX_exp_indirect_cost.'
	assert np.any(temp_fX_expected_cost == 0) == False, 'Error in correcting fX_expected_cost.'


	# # print the expected network performance
	# if print_results:
	# 	print 'for R = ', n_retrofits, ' expected travel times = ', f_X_avg_time, f_X_avg_time-tt0, alpha*(f_X_avg_time-tt0)/3600
	# 	print 'for R = ', n_retrofits, ' expected trips made = ', f_X_avg_trip, trips0-f_X_avg_trip, beta*(trips0-f_X_avg_trip)
	# 	print 'for R = ', n_retrofits, ' expected indirect costs = ', f_X_exp_indirect_cost*24*125 #24*125*(alpha*(f_X_avg_time-tt0)/3600 + beta*(trips0-f_X_avg_trip))
	# 	# print 24*125*(alpha*(f_X_avg_time-tt0)/3600 + beta*(trips0-f_X_avg_trip)) # should be the same as f_X_exp_indirect_cost
	# 	print 'for R = ', n_retrofits, ' expected direct costs = ', f_X_exp_direct_cost
	# 	# print f_X_exp_direct_cost + (f_X_exp_cost-f_X_exp_direct_cost)
	# 	print 'for R = ', n_retrofits, ' expected total cost = ', f_X_exp_cost #, f_X_exp_indirect_cost*24*125+f_X_exp_direct_cost


	# return f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, f_X_exp_indirect_cost*24*125, f_X_exp_direct_cost, f_X_exp_cost
	return temp_fX_avg_times, temp_fX_avg_vmts, temp_fX_avg_trips, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, temp_fX_exp_indirect_cost*24*125, temp_fX_exp_direct_cost, temp_fX_expected_cost


def get_all_sobol_retrofit_result(strategy):

	baseline = 32417786.20037872 # for S = 45, expected total cost with REVISED AVERAGE, not including retrofit cost

	print_results = False

	n_scenarios = 45
	filename = '_sf_fullr'


	if strategy == 'Sobol exp. total':
		# Sobol-index based retrofit strategy
		n_batches = 10
		batch_size = 7
		results = np.zeros(n_batches*batch_size,)
		for i in range(0,n_batches):
			output_folder = 'sobol_output/retrofits/sobol_retrofits/sobol_'+str(i)+'/' # before correcting expectation computation
			# output_folder = 'sobol_output/retrofits/ret_revised_avg/r' + str(i) + '/' # ordering based on corrected expectation computation
			f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, \
			f_X_exp_indirect_cost, f_X_exp_direct_cost, f_X_exp_cost = get_retrofit_results(output_folder, n_scenarios,
																								   filename,print_results)

			results[i*batch_size:(i+1)*batch_size] = f_X_exp_cost

	# 	reduction = [(r-baseline)/baseline for r in results]
	# else:
	# 	reduction = None

	return results

def plot_retrofit_results(positive=True, training=False): # MOST UPDATED PLOTTING OF EXP. TOTAL COST REDUCTION VS. NUMBER OF RETROFITS
	# folder = 'figs_diff_p/'
	# folder = 'figs_paper_final_redone/'
	folder = 'figs/'
	# if training:
	# 	subfolder = 'training/'
	# else:
	# 	subfolder = 'testing/'

	# fig_folder = folder + subfolder
	fig_folder = folder

	no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = precompute_network_performance()

	# Get baseline, i.e. result when R = 0.
	baseline = get_baseline_retrofit_results(n_retrofits=0, no_damage_travel_time=no_damage_travel_time, no_damage_vmt=no_damage_vmt, no_damage_trips_made=no_damage_trips_made, training=training, detailed=False)
	# Get best case, i.e. result when R = 71. (all bridges retrofitted)
	best = get_baseline_retrofit_results(n_retrofits=71, no_damage_travel_time=no_damage_travel_time, no_damage_vmt=no_damage_vmt, no_damage_trips_made=no_damage_trips_made, training=training, detailed=False)

	# first, get all the data for the other strategies -- these are for S = 45
	strategies = ['oldest', 'busiest', 'weakest', 'composite', 'OAT total', 'Sobol exp. total']
	# strategies = [ 'Sobol exp. total']

	# n_retrofits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 71]
	n_retrofits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 71]
	R = len(n_retrofits)

	results = {}
	for s in strategies:
		print 'Starting strategy: ', s
		if s == 'Sobol exp. total':
			temp = get_all_sobol_retrofit_result(s)
			temp = list(temp)
			temp_results = [baseline] + temp
			temp_results.append(best)
			# filter to get just results corresponding to n_retrofits, the numbers of retrofits of interest
			final_results = [temp_results[r] for r in n_retrofits]
			results[s] = [t for t in final_results]
			print len(results[s]), results[s]
			# get just the resuts corresponding to n_retrofits, the numbers of retrofits of interest

		else:
			temp_results = np.zeros(R, )  # holder for max cost at each n_retrofit for strategy s
			i = 0
			for r in n_retrofits:
				if r == 0:
					temp_results[i] = baseline
				elif r == 71:
					temp_results[i] = best
				else:
					# _, _, indirect_costs, direct_costs, expected_total_cost = get_retrofit_results_from_file(r, strategy=s,
					# 																						 training=training)
					expected_total_cost = get_retrofit_results_from_file(r, strategy=s, no_damage_travel_time=no_damage_travel_time, no_damage_vmt=no_damage_vmt, no_damage_trips_made=no_damage_trips_made,training=training)
					# total_costs = get_total_costs(indirect_costs, direct_costs)
					temp_results[i] = expected_total_cost  # expected total cost for retrofit strategy s and r retrofits
				i += 1
			print s, len(temp_results)
			results[s] = [t for t in temp_results]

	percent_results = {}
	# convert max total costs to percent reduction in max total cost of network performance relative to R = 0
	for s in strategies:
		temp_results = np.zeros(R, )
		for r in range(0, R):
			temp_results[r] = compute_percent_reduction(baseline, results[s][r])
		percent_results[s] = [t for t in temp_results]
		print s, percent_results[s]

	# if positive, convert percent reductions into percent gains
	if positive:
		for s in strategies:
			for i in range(0,R):
				percent_results[s][i] = percent_results[s][i] * -1
		# results = results*-1

	# exp. total cost vs. number of bridges retrofitted for each strategy
	if positive:
		title = 'sf_fullr_exp_total_cost_vs_n_bridges_positive'
	else:
		title = 'sf_fullr_exp_total_cost_vs_n_bridges'

	# with open(fig_folder + 'percent_results.pkl', 'wb') as f:
	# 	pickle.dump(percent_results,f)
	#
	# with open(fig_folder + 'percent_results.pkl', 'rb') as f:
	# 	percent_results = pickle.load(f)

	marker_style = 'o'
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(n_retrofits, percent_results['oldest'], color=get_color('oldest'), marker=marker_style, label='age')
	ax.plot(n_retrofits, percent_results['busiest'], color=get_color('busiest'), marker=marker_style,
			label='traffic volume')
	ax.plot(n_retrofits, percent_results['weakest'], color=get_color('weakest'), marker=marker_style, label='fragility')
	ax.plot(n_retrofits, percent_results['composite'], color=get_color('composite'), marker=marker_style,
			label='composite')
	ax.plot(n_retrofits, percent_results['OAT total'], color=get_color('OAT'), marker=marker_style, label='OAT')
	ax.plot(n_retrofits, percent_results['Sobol exp. total'], color=get_color('Sobol'), marker=marker_style,
			label='Sobol, $\\mathbb{E}[C]$')
	# ax.plot(n_retrofits, results, marker=marker_style, ls='--', color=get_color('Sobol'), label='Sobol, $P_{98.5}$')
	ax.plot(n_retrofits, [percent_results['oldest'][-1] for i in n_retrofits], color='black', label='all retrofitted')
	ax.set_xlim([0, 71])
	# plt.legend(loc='best', prop={'size': 10})
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	handles, labels = ax.get_legend_handles_labels()
	handles = [handles[6], handles[5], handles[4], handles[0], handles[2], handles[3], handles[1]]
	labels = [labels[6], labels[5], labels[4], labels[0], labels[2], labels[3], labels[1]]
	ax.legend(handles, labels, loc='best', frameon=False)
	plt.xlabel('Number of retrofitted bridges, $R$')
	plt.ylabel('$\\%$ reduction in $\\mathbb{E}[C]$  of network performance')
	plt.savefig(fig_folder + title + '.png', bbox_inches='tight')

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(n_retrofits, percent_results['oldest'], color=get_color('oldest'), marker=marker_style, label='age')
	ax.plot(n_retrofits, percent_results['busiest'], color=get_color('busiest'), marker=marker_style,
			label='traffic volume')
	ax.plot(n_retrofits, percent_results['weakest'], color=get_color('weakest'), marker=marker_style, label='fragility')
	ax.plot(n_retrofits, percent_results['composite'], color=get_color('composite'), marker=marker_style,
			label='composite')
	ax.plot(n_retrofits, percent_results['OAT total'], color=get_color('OAT'), marker=marker_style, label='OAT')
	ax.plot(n_retrofits, percent_results['Sobol exp. total'], color=get_color('Sobol'), marker=marker_style,
			label='Sobol, $\\mathbb{E}[C]$')
	# ax.plot(n_retrofits, results, marker=marker_style, ls='--', color=get_color('Sobol'), label='Sobol, $P_{98.5}$')
	ax.plot(n_retrofits, [percent_results['oldest'][-1] for i in n_retrofits], color='black', label='all retrofitted')
	ax.set_xlim([0, 10])
	# plt.legend(loc='best', prop={'size': 10})
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	handles, labels = ax.get_legend_handles_labels()

	handles = [handles[6], handles[5], handles[4], handles[0], handles[2], handles[3], handles[1]]
	labels = [labels[6], labels[5], labels[4], labels[0], labels[2], labels[3], labels[1]]

	ax.legend(handles, labels, loc='upper left', bbox_to_anchor = (0.01, 0.3, 0.5, 0.5), frameon=False)
	plt.xlabel('Number of retrofitted bridges, $R$')
	plt.ylabel('$\\%$ reduction in $\\mathbb{E}[C]$  of network performance')
	plt.savefig(fig_folder + title + '_zoom.png', bbox_inches='tight')

	plt.show()

plot_retrofit_results()