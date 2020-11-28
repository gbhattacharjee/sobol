from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from process_results_sf import get_sf_fullr_dict
import numpy as np
import compute_bridge_sobol_sf_full as cbs
import os, pickle, zipfile, shutil, util
from math import log

alpha = 48 # dollars per hour
beta = 78*8 # dollars per hour times hours

def compute_weighted_average_performance(lnsas, map_weights, num_damage_maps, travel_times, vmts, trips_made,
										 no_damage_travel_time, no_damage_vmt, no_damage_trips_made, direct_costs):
	# Compute weighted average of performance metrics for a single sample of a fragility function vector.

	scenarios = len(lnsas) # number of scenarios under consideration

	# GB ADDITION -- computed weighted average (expectation) of travel time and other metrics of interest
	average_travel_time = 0
	average_trips_made = 0
	average_vmt = 0
	average_direct_costs = 0

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


def get_results_from_pickles(results_directory, results_folder_stub, n_batches, max_batch=20, scenarios=30,
							 cost='total', retrofit=True, p=False, first_order=False, batch_size=10):

	print 'In function, cwd is : ', os.getcwd()
	S = scenarios
	D = 10

	bridge_dict, bridge_ids = get_sf_fullr_dict()

	n_bridges = len(bridge_ids)  # how many bridges we considered
	n_samples = n_batches * batch_size  # how many samples of the fragility function parameters we used

	# create placeholders in which we'll store the real f_V and f_X values
	f_X_times = np.zeros((n_samples, S * D))
	f_X_trips = np.zeros((n_samples, S * D))
	f_X_vmts = np.zeros((n_samples, S * D))
	f_X_delay_costs = np.zeros((n_samples, S * D))
	f_X_conn_costs = np.zeros((n_samples, S * D))
	f_X_indirect_costs = np.zeros((n_samples, S * D))
	f_X_direct_costs = np.zeros((n_samples, S * D))

	f_X_avg_time = np.zeros((n_samples,))
	f_X_avg_trip = np.zeros((n_samples,))
	f_X_exp_indirect_cost = np.zeros((n_samples,))
	f_X_exp_direct_cost = np.zeros((n_samples,))
	f_X_exp_cost = np.zeros((n_samples,))  # total expected cost
	f_X_ret_cost = np.zeros((n_samples,))  # retrofit cost (deterministic)
	f_X_avg_vmt = np.zeros((n_samples,))

	f_V_times = np.zeros((n_samples, scenarios * D, n_bridges))
	f_V_trips = np.zeros((n_samples, scenarios * D, n_bridges))
	f_V_vmts = np.zeros((n_samples, scenarios * D, n_bridges))

	f_V_avg_time = np.zeros((n_samples, n_bridges))
	f_V_avg_trip = np.zeros((n_samples, n_bridges))
	f_V_avg_vmt = np.zeros((n_samples, n_bridges))

	f_V_delay_costs = np.zeros((n_samples, scenarios * D, n_bridges))
	f_V_conn_costs = np.zeros((n_samples, scenarios * D, n_bridges))
	f_V_indirect_costs = np.zeros((n_samples, scenarios * D, n_bridges))
	f_V_direct_costs = np.zeros((n_samples, scenarios * D, n_bridges))
	f_V_exp_indirect_cost = np.zeros((n_samples, n_bridges))
	f_V_exp_direct_cost = np.zeros((n_samples, n_bridges))
	f_V_exp_cost = np.zeros((n_samples, n_bridges))
	f_V_ret_cost = np.zeros((n_samples, n_bridges))

	directory = results_directory
	filename = '_sf_fullr.pkl'

	map_indices_input = 'sobol_input/sf_fullr_training_map_indices.pkl'  # S = 30 for training sf_fullr
	map_weights_input = 'sobol_input/sf_fullr_training_map_weights.pkl'  # S = 30 for training sf_fullr

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

	no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = cbs.precompute_network_performance()

	j = 0
	skipped = 0
	for i in range(0, max_batch):

		folder = results_folder_stub + str(i) + '/'

		# if not p:
		# 	filename = '_cs' + str(cs_number) + '.pkl'
		# else:
		# 	filename = '_cs' + str(cs_number) + 'p.pkl'

		#
		# # Declare input file names -- from binary distribution
		# F_input = 'sobol_input/F_samples_sft2_new.pkl'  # N = 200
		# F_prime_input = 'sobol_input/F_prime_samples_sft2_new.pkl'  # N = 200

		# declare output file names
		fX_times_output = directory + folder + 'fX_times' + filename  # travel times for f_X
		fX_trips_output = directory + folder + 'fX_trips' + filename  # trips made for f_X
		fX_vmts_output = directory + folder + 'fX_vmts' + filename  # VMTs for f_X
		fX_avg_times_output = directory + folder + 'fX_avg_time' + filename  # average TT
		fX_avg_trips_output = directory + folder + 'fX_avg_trips' + filename  # average trips made
		fX_avg_vmts_output = directory + folder + 'fX_avg_vmts' + filename  # average VMT
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

		damage_x_output = directory + folder + 'damage_x' + filename
		damage_v_output = directory + folder + 'damage_v' + filename
		S_cost_output = directory + folder + 'S_cost' + filename
		tau_cost_output = directory + folder + 'tau_cost' + filename

		sobol_index_dict_output = directory + folder + 'sobol_dict' + filename

		try:  # in case not all batches were completed

			extracted_file = results_directory + 'run_sf_' + str(i) + '.zip'
			# # print 'trying extracted file: ', extracted_file
			# with zipfile.ZipFile(extracted_file, 'r') as zip_ref:
			# 	zip_ref.extractall(results_directory)

			if i < 80:  # TODO -- fix this -- not sure why I'm getting different behavior for same function.
				with zipfile.ZipFile(extracted_file, 'r') as zip_ref:
					zip_ref.extractall(results_directory)
			else:
				with zipfile.ZipFile(extracted_file, 'r') as zip_ref:
					zip_ref.extractall(results_directory + folder)

			with open(fX_times_output, 'rb') as f:
				temp_fX_times = pickle.load(f)
			with open(fX_trips_output, 'rb') as f:
				temp_fX_trips = pickle.load(f)
			with open(fX_vmts_output, 'rb') as f:
				temp_fX_vmts = pickle.load(f)
			# with open(fX_avg_times_output, 'rb') as f:
			# 	temp_fX_avg_times = pickle.load(f)
			# with open(fX_avg_trips_output, 'rb') as f:
			# 	temp_fX_avg_trips = pickle.load(f)
			# with open(fX_avg_vmts_output, 'rb') as f:
			# 	temp_fX_avg_vmts = pickle.load(f)
			# with open(fX_delay_costs_output, 'rb') as f:
			# 	temp_fX_delay_costs = pickle.load(f)
			# with open(fX_conn_costs_output, 'rb') as f:
			# 	temp_fX_conn_costs = pickle.load(f)
			# with open(fX_indirect_costs_output, 'rb') as f:
			# 	temp_fX_indirect_costs = pickle.load(f)
			with open(fX_direct_costs_output, 'rb') as f:
				temp_fX_direct_costs = pickle.load(f)
			# with open(fX_exp_indirect_cost_output, 'rb') as f:
			# temp_fX_exp_indirect_cost = pickle.load(f)
			# with open(fX_exp_direct_cost_output, 'rb') as f:
			# temp_fX_exp_direct_cost = pickle.load(f)
			# with open(fX_expected_cost_output,'rb') as f:
			# temp_fX_expected_cost = pickle.load(f)
			with open(fX_retrofit_cost_output, 'rb') as f:
				temp_fX_retrofit_cost = pickle.load(f)

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
																			 travel_times=temp_fX_times[k, :],
																			 vmts=temp_fX_vmts[k, :],
																			 trips_made=temp_fX_trips[k, :],
																			 no_damage_travel_time=no_damage_travel_time,
																			 no_damage_vmt=no_damage_vmt,
																			 no_damage_trips_made=no_damage_trips_made,
																			 direct_costs=temp_fX_direct_costs[k, :])

				temp_fX_avg_times[k] = average_travel_time
				temp_fX_avg_vmts[k] = average_vmt
				temp_fX_avg_trips[k] = average_trips_made
				temp_fX_exp_direct_cost[k] = average_direct_cost
				temp_fX_exp_indirect_cost[k] = average_indirect_cost  # hourly
				temp_fX_expected_cost[k] = 24 * 125 * average_indirect_cost + average_direct_cost

			assert np.any(temp_fX_exp_indirect_cost == 0) == False, 'Error in correcting fX_exp_indirect_cost.'
			assert np.any(temp_fX_expected_cost == 0) == False, 'Error in correcting fX_expected_cost.'

			f_X_times[j * batch_size:(j + 1) * batch_size, ] = temp_fX_times
			f_X_trips[j * batch_size:(j + 1) * batch_size, ] = temp_fX_trips
			f_X_vmts[j * batch_size:(j + 1) * batch_size, ] = temp_fX_vmts
			f_X_avg_time[j * batch_size:(j + 1) * batch_size, ] = temp_fX_avg_times
			f_X_avg_trip[j * batch_size:(j + 1) * batch_size, ] = temp_fX_avg_trips
			f_X_avg_vmt[j * batch_size:(j + 1) * batch_size, ] = temp_fX_avg_vmts
			# f_X_delay_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_delay_costs
			# f_X_conn_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_conn_costs
			# f_X_indirect_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_indirect_costs
			f_X_direct_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_direct_costs
			f_X_exp_indirect_cost[j * batch_size:(
															 j + 1) * batch_size, ] = 24 * 125 * temp_fX_exp_indirect_cost  # TODO: assume 125 days of closure, 24 hours per day
			f_X_exp_direct_cost[j * batch_size:(j + 1) * batch_size, ] = temp_fX_exp_direct_cost
			# f_X_exp_cost[j * batch_size:(j + 1) * batch_size, ] = temp_fX_expected_cost
			f_X_exp_cost[j * batch_size:(
													j + 1) * batch_size, ] = 24 * 125 * temp_fX_exp_indirect_cost + temp_fX_exp_direct_cost  # TODO: assume 125 days of closure, 24 hours per day
			f_X_ret_cost[j * batch_size:(j + 1) * batch_size, ] = temp_fX_retrofit_cost

			with open(fV_times_output, 'rb') as f:
				temp_fV_times = pickle.load(f)
			with open(fV_trips_output, 'rb') as f:
				temp_fV_trips = pickle.load(f)
			with open(fV_vmts_output, 'rb') as f:
				temp_fV_vmts = pickle.load(f)
			with open(fV_avg_times_output, 'rb') as f:
				temp_fV_avg_times = pickle.load(f)
			# with open(fV_avg_trips_output, 'rb') as f:
			# 	temp_fV_avg_trips = pickle.load(f)
			# with open(fV_avg_vmts_output, 'rb') as f:
			# 	temp_fV_avg_vmts = pickle.load(f)
			with open(fV_delay_costs_output, 'rb') as f:
				temp_fV_delay_costs = pickle.load(f)
			with open(fV_conn_costs_output, 'rb') as f:
				temp_fV_conn_costs = pickle.load(f)
			with open(fV_indirect_costs_output, 'rb') as f:
				temp_fV_indirect_costs = pickle.load(f)
			with open(fV_direct_costs_output, 'rb') as f:
				temp_fV_direct_costs = pickle.load(f)
			# with open(fV_exp_indirect_cost_output, 'rb') as f:
			# 	temp_fV_exp_indirect_cost = pickle.load(f)
			# with open(fV_exp_direct_cost_output, 'rb') as f:
			# 	temp_fV_exp_direct_cost = pickle.load(f)
			# with open(fV_expected_cost_output,'rb') as f:
			# 	temp_fV_expected_cost = pickle.load(f)
			with open(fV_retrofit_cost_output, 'rb') as f:
				temp_fV_retrofit_cost = pickle.load(f)

			# print 'temp_fV_avg_times.shape, temp_fV_times.shape', temp_fV_avg_times.shape, temp_fV_times.shape
			# print 'temp_fV_direct_costs.shape = ', temp_fV_direct_costs.shape

			temp_fV_avg_times = np.zeros((batch_size, n_bridges))
			temp_fV_avg_vmts = np.zeros((batch_size, n_bridges))
			temp_fV_avg_trips = np.zeros((batch_size, n_bridges))
			temp_fV_exp_indirect_cost = np.zeros((batch_size, n_bridges))
			temp_fV_exp_direct_cost = np.zeros((batch_size, n_bridges))
			temp_fV_expected_cost = np.zeros((batch_size, n_bridges))

			for k in range(0, batch_size):
				for l in range(0, n_bridges):
					# print '*** batch = ', i, ' sample = ', k
					average_travel_time, average_vmt, average_trips_made, average_direct_cost, average_delay_cost, average_connectivity_cost, \
					average_indirect_cost = compute_weighted_average_performance(lnsas, map_weights, num_damage_maps=10,
																				 travel_times=temp_fV_times[k, :, l],
																				 vmts=temp_fV_vmts[k, :, l],
																				 trips_made=temp_fV_trips[k, :, l],
																				 no_damage_travel_time=no_damage_travel_time,
																				 no_damage_vmt=no_damage_vmt,
																				 no_damage_trips_made=no_damage_trips_made,
																				 direct_costs=temp_fV_direct_costs[k, :,
																							  l])  # TODO -- make sure all slices are correct!!!!

					temp_fV_avg_times[k, l] = average_travel_time
					temp_fV_avg_vmts[k, l] = average_vmt
					temp_fV_avg_trips[k, l] = average_trips_made
					temp_fV_exp_direct_cost[k, l] = average_direct_cost
					temp_fV_exp_indirect_cost[k, l] = average_indirect_cost  # hourly
					temp_fV_expected_cost[k, l] = 24 * 125 * average_indirect_cost + average_direct_cost

			assert np.any(temp_fV_exp_indirect_cost == 0) == False, 'Error in correcting fV_exp_indirect_cost.'
			assert np.any(temp_fV_expected_cost == 0) == False, 'Error in correcting fV_expected_cost.'

			f_V_times[j * batch_size:(j + 1) * batch_size, :] = temp_fV_times
			f_V_trips[j * batch_size:(j + 1) * batch_size, :] = temp_fV_trips
			f_V_vmts[j * batch_size:(j + 1) * batch_size, :] = temp_fV_vmts
			f_V_avg_time[j * batch_size:(j + 1) * batch_size, :] = temp_fV_avg_times
			f_V_avg_trip[j * batch_size:(j + 1) * batch_size, :] = temp_fV_avg_trips
			f_V_avg_vmt[j * batch_size:(j + 1) * batch_size, :] = temp_fV_avg_vmts
			f_V_delay_costs[j * batch_size:(j + 1) * batch_size, :] = temp_fV_delay_costs
			f_V_conn_costs[j * batch_size:(j + 1) * batch_size, :] = temp_fV_conn_costs
			f_V_indirect_costs[j * batch_size:(j + 1) * batch_size, :] = temp_fV_indirect_costs
			f_V_direct_costs[j * batch_size:(j + 1) * batch_size, :] = temp_fV_direct_costs
			f_V_exp_indirect_cost[j * batch_size:(j + 1) * batch_size,
			:] = 24 * 125 * temp_fV_exp_indirect_cost  # TODO: assume 125 days of closure, 24 hours per day
			f_V_exp_direct_cost[j * batch_size:(j + 1) * batch_size, :] = temp_fV_exp_direct_cost
			# f_V_exp_cost[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_expected_cost
			f_V_exp_cost[j * batch_size:(j + 1) * batch_size,
			:] = 24 * 125 * temp_fV_exp_indirect_cost + temp_fV_exp_direct_cost  # TODO: assume 125 days of closure, 24 hours per day
			f_V_ret_cost[j * batch_size:(j + 1) * batch_size, :] = temp_fV_retrofit_cost

			j += 1

			shutil.rmtree(results_directory + 'run_sf_' + str(i))

		except:
			print 'skipped f_X and f_V for batch ', i, 'of ', n_batches, folder, directory + folder + 'fX_times' + filename
			skipped += 1

			try:
				shutil.rmtree(extracted_file)
			except:
				pass

	print 'skipped ', skipped, ' of ', max_batch, ' batches'

	if not first_order:  # i.e., if total-order Sobol' indices
		if retrofit:
			print 'here here here'
			# print f_X_exp_direct_cost.shape, f_X_ret_cost.shape
			if cost == 'total':
				temp = f_X_exp_cost + f_X_ret_cost
				print f_X_exp_cost[0], f_X_ret_cost[0], temp[0]
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_cost + f_X_ret_cost,
																 f_V_exp_cost + f_V_ret_cost,
																 normalize=True)  # was originally exp_cost - retrofit_cost
			elif cost == 'indirect':
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_indirect_cost + f_X_ret_cost,
																 f_V_exp_indirect_cost + f_V_ret_cost, normalize=True)
			elif cost == 'direct':
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_direct_cost + f_X_ret_cost,
																 f_V_exp_direct_cost + f_V_ret_cost, normalize=True)
		else:
			if cost == 'total':
				print '*** correct setting for Sobol indices based on expected total cost, not including retrofit cost'
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_cost, f_V_exp_cost, normalize=True)
			elif cost == 'indirect':
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_indirect_cost, f_V_exp_indirect_cost,
																 normalize=True)
			elif cost == 'direct':
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_direct_cost, f_V_exp_direct_cost,
																 normalize=True)
	else:
		if retrofit:
			# print f_X_exp_direct_cost.shape, f_X_ret_cost.shape
			if cost == 'total':
				print 'right here'
				f_X_exp_cost = f_X_exp_direct_cost + f_X_exp_indirect_cost
				f_V_exp_cost = f_V_exp_direct_cost + f_V_exp_indirect_cost
				print 'sum of f_X_exp_cost in process results ', sum(f_X_exp_cost), sum(f_X_exp_direct_cost), sum(
					f_X_exp_indirect_cost)
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_cost, f_V_exp_cost, normalize=True)
			# S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_cost-f_X_ret_cost, f_V_exp_cost-f_V_ret_cost, normalize=True)
			elif cost == 'indirect':
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_indirect_cost - f_X_ret_cost,
																	   f_V_exp_indirect_cost - f_V_ret_cost,
																	   normalize=True)
			elif cost == 'direct':
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_direct_cost - f_X_ret_cost,
																	   f_V_exp_direct_cost - f_V_ret_cost,
																	   normalize=True)
		else:
			if cost == 'total':
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_cost, f_V_exp_cost, normalize=True)
			elif cost == 'indirect':
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_indirect_cost, f_V_exp_indirect_cost,
																	   normalize=True)
			elif cost == 'direct':
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_direct_cost, f_V_exp_direct_cost,
																	   normalize=True)

	print 'for ', cost, ' cost-based Sobol indices, sum = ', sum(S_exp_cost)

	sobol_index_dict = {}
	i = 0
	for b in bridge_ids:
		sobol_index_dict[b] = S_exp_cost[i]
		# print 'b = ', b, ' sobol_index_dict[b] = ', sobol_index_dict[b]
		print b, sobol_index_dict[b]
		i += 1

	return sobol_index_dict, f_X_exp_cost, f_V_exp_cost, f_X_ret_cost, f_V_ret_cost


def plot_convergence_all_bridges():
	results_directory = 'sobol_output/run_sf_fullr_total_all/'
	results_folder_stub = 'run_sf_'
	n_batches =  92 # originally 37; then added 20 but lost 3 due to time-out; MAX OF 92
	max_batches = 140
	n_scenarios = 30 # S= 30 for training, S = 45 for testing
	print '****** sf_fullr results ******'
	sobol_index_dict_cost, f_X_cost, f_V_cost, _, _ = get_results_from_pickles(results_directory, results_folder_stub, n_batches, max_batch=max_batches, scenarios=n_scenarios, cost='total', retrofit=False, batch_size=5) # total cost, not including retrofit cost


	figures_folder = 'figs/convergence/' # TODO -- change this to wherever you'd like all of the plots to be saved

	bridge_dict, bridges = get_sf_fullr_dict()
	n_samples = f_X_cost.shape[0]
	n_bridges = len(bridges)
	figure_N_testing = 'N_test_'
	figure_CI_size = 'CI_size_'
	figure_N_conv = 'N_convergence_'
	bridge_number = 'bridge_'

	# plot settings -- small titles
	params = {'axes.titlesize':'small',}
	plb.rcParams.update(params)

	M = np.arange(20, n_samples + 1, 10)
	n_sizes = len(M)
	S_subset = np.zeros((n_sizes,n_bridges))
	CI_subset = np.zeros((n_sizes,n_bridges))
	mean_sobol = np.ones((n_bridges,))

	# plot theoretical convergence of Monte Carlo for size of 95% CI
	conv_theo_y = np.asarray([float(1/np.sqrt(M[i])) for i in range(0,n_sizes)])

	print 'M', M
	print 'conv_theo_y', conv_theo_y

	b = 0
	for bridge in bridges:

		bridge_index = b # DEFINE the index of the bridge you want to consider
		if sobol_index_dict_cost[bridge] > 0:
			N = f_X_cost.shape[0] # total number of samples we have access to
			n_indices = np.arange(0,N,step=1)

			M = np.arange(10,n_samples + 1,10)
			n_sizes = len(M)
			n_trials = 100 # how many times we should choose a random subset of size M from the total N samples

			tau_hat = np.empty((n_sizes,n_trials))
			S_hat = np.empty((n_sizes,n_trials))

			# compute approximations of the 'true' value of the Sobol index for this bridge

			for i in range(0,n_sizes): # for each subset size
				for j in range(0,n_trials): # for each trial
					# randomly choose m samples of f_X and f_V
					indices = np.random.choice(n_indices, size=(M[i],), replace=True)
					temp_fX = f_X_cost[indices,]
					temp_fV = f_V_cost[indices,bridge_index] # second index should be that of the bridge of interest

					# Total-order approximation
					Sigma = 0
					for k in range(0, M[i]):
						Sigma += (temp_fX[k] - temp_fV[k]) ** 2

					temp_tau = Sigma / (2 * M[i])

					var_hat = cbs.compute_sample_variance(temp_fX)

					temp_S = temp_tau/var_hat

					tau_hat[i][j] = temp_tau
					S_hat[i][j] = temp_S

					#print 'Estimated Sobol index with ', M[i] ,' samples: ', temp_tau, temp_S

			mean_tau_hat = np.average(tau_hat,axis=1)
			mean_S_hat = np.average(S_hat,axis=1)
			var_S_hat = np.asarray([cbs.compute_sample_variance(S_hat[i,:]) for i in range(0,n_sizes)])
			CI_S_hat = np.asarray([cbs.compute_CI(S_hat[i,:],confidence=0.95) for i in range(0,n_sizes)])
			diff_S_hat = abs(mean_S_hat - sobol_index_dict_cost[bridge])
			#
			# print 'mean_S_hat', mean_S_hat.shape
			# print 'CI_S_hat', CI_S_hat.shape

			S_subset[:,b] = mean_S_hat
			CI_subset[:,b] = CI_S_hat # 95% CI on each bridge's total-order Sobol index

			# PLOT the size of the 95% confidence interval on the estimated Sobol index vs. the number of samples M used to compute the index
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.set_xlabel('$n_{samples}$')
			ax.set_ylabel('Size of the 95% CI on the est. total-order Sobol index')
			ax.loglog(M, 2*abs(CI_S_hat), marker = 'o')
			ax.loglog(M, conv_theo_y, ls='--', color='m', label='theoretical MC convergence')
			ax.set_xlim([0,max(M)+100])
			ax.tick_params(axis='both', which='major', labelsize='small')
			ax.legend(loc='best',prop=dict(size=10))
			ax.set_title('Bridge ' + bridge + ': 95% CI size vs. sample size')
			plt.savefig(figures_folder + figure_CI_size + bridge_number + bridge, bbox_inches='tight')

			# PLOT difference between 'true' and mean approximate Sobol index value as M increases
			fig1 = plt.figure()
			ax1 = fig1.add_subplot(111)
			ax1.set_xlabel('$n_{samples}$')
			ax1.set_ylabel('$|\\hat{\\bar{S}}_b^2 - \\hat{\\mu}_{\\hat{\\bar{S}}_b^2}|$')
			ax1.loglog(M, diff_S_hat, marker = 'o')
			ax1.tick_params(axis='both', which='major', labelsize='small')
			ax1.set_title('Bridge ' + bridge + ': absolute error vs. sample size')
			plt.savefig(figures_folder + figure_N_conv + bridge_number + bridge, bbox_inches='tight')

			# PLOT m (size of subset) vs. Sobol index approximation
			fig_m_1 = plt.figure()
			ax1 = fig_m_1.add_subplot(111)
			ax1.set_xlabel('$n_{samples}$')
			ax1.set_ylabel('$\\hat{\\bar{S}}_b^2$')
			ax1.errorbar(M, mean_S_hat, xerr = None, yerr=CI_S_hat, marker='o', ls='none', label='$\\hat{\\mu}_{\\hat{\\bar{S}}_b^2}$ for $100$ trials')
			ax1.axhline(sobol_index_dict_cost[bridge], c='m', label= '$\\hat{\\bar{S}}_b^2$ with $n_{samples} = 185$') # plot the 'true' value of the Sobol index for this bridge
			ax1.set_xlim([0,max(M)+10])
			ax1.tick_params(axis='both', which='major', labelsize='small')
			ax1.legend(loc='best', numpoints=1)
			ax1.set_title('Bridge ' + bridge + ': est. total-order Sobol index vs. sample size')
			plt.savefig(figures_folder + figure_N_testing + bridge_number + bridge, bbox_inches='tight')

		b += 1

def plot_convergence_important_bridges(top5=True):
	# plot the convergence of the top 3 bridges to check whether they are distinguishable

	results_directory = 'sobol_output/run_sf_fullr_total_all/'
	results_folder_stub = 'run_sf_'
	n_batches = 92
	max_batches = 140
	n_scenarios = 30  # S= 30 for training, S = 45 for testing
	print '****** sf_fullr results ******'
	sobol_index_dict_cost, f_X_cost, f_V_cost, _, _ = get_results_from_pickles(results_directory, results_folder_stub,
																			   n_batches, max_batch=max_batches,
																			   scenarios=n_scenarios, cost='total',
																			   retrofit=False,
																			   batch_size=5)  # total cost, not including retrofit cost

	figures_folder = 'figs/'

	bridge_dict, bridges = get_sf_fullr_dict()
	n_samples = f_X_cost.shape[0]
	n_bridges = len(bridges)

	if top5:
		important_bridges = [1066, 976, 956, 1027, 928]
	else:
		important_bridges = [1027, 928, 977] # ranked 4, 5, 6

	# important_bridges_labels = ['4th', '5th', '6th', '7th', '8th', '9th']
	important_bridges = [str(b) for b in important_bridges]

	bridge_index_lookup = {}

	# colors = ['magenta', 'cyan', 'blue', 'green', 'yellow']
	R = len(important_bridges)
	M = np.arange(10, n_samples + 1, 10)
	n_sizes = len(M)
	b = 0
	mean_tau_hat = np.zeros((len(M),n_bridges))
	mean_S_hat = np.zeros((len(M),n_bridges))
	var_S_hat = np.zeros((len(M),n_bridges))
	CI_S_hat = np.zeros((len(M),n_bridges))
	diff_S_hat = np.zeros((len(M),n_bridges))

	for bridge in bridges:
		if bridge in important_bridges:
			bridge_index = b  # DEFINE the index of the bridge you want to consider
			bridge_index_lookup[bridge] = bridge_index
			if sobol_index_dict_cost[bridge] > 0:
				N = f_X_cost.shape[0]  # total number of samples we have access to
				n_indices = np.arange(0, N, step=1)

				# M = np.arange(20, n_samples + 1, 10)
				# n_sizes = len(M)
				n_trials = 100  # how many times we should choose a random subset of size M from the total N samples

				tau_hat = np.empty((n_sizes, n_trials))
				S_hat = np.empty((n_sizes, n_trials))

				# compute approximations of the 'true' value of the Sobol index for this bridge

				for i in range(0, n_sizes):  # for each subset size
					for j in range(0, n_trials):  # for each trial
						# randomly choose m samples of f_X and f_V
						indices = np.random.choice(n_indices, size=(M[i],), replace=True)
						temp_fX = f_X_cost[indices,]
						temp_fV = f_V_cost[indices, bridge_index]  # second index should be that of the bridge of interest

						# Total-order approximation
						Sigma = 0
						for k in range(0, M[i]):
							Sigma += (temp_fX[k] - temp_fV[k]) ** 2

						temp_tau = Sigma / (2 * M[i])

						var_hat = cbs.compute_sample_variance(temp_fX)

						# print 'temp_fX = ', temp_fX

						temp_S = temp_tau / var_hat

						tau_hat[i][j] = temp_tau
						S_hat[i][j] = temp_S

				# print 'Estimated Sobol index with ', M[i] ,' samples: ', temp_tau, temp_S
				mean_tau_hat[:,b] = np.average(tau_hat, axis=1)
				mean_S_hat[:,b] = np.average(S_hat, axis=1)
				var_S_hat[:,b] = np.asarray([cbs.compute_sample_variance(S_hat[i, :]) for i in range(0, n_sizes)])
				CI_S_hat[:,b] = np.asarray([cbs.compute_CI(S_hat[i, :], confidence=0.99) for i in range(0, n_sizes)])
				diff_S_hat[:,b] = abs(mean_S_hat[:,b] - sobol_index_dict_cost[bridge])

		b+= 1


	# colors= ['black','#a50f15', '#de2d26', '#fb6a4a', '#fc9272', '#fcbba1'] # for final paper
	colors = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15','#67000d'] # top 9
	colors.reverse()

	# PLOT m (size of subset) vs. Sobol index approximation
	fig_m_1 = plt.figure()
	ax1 = fig_m_1.add_subplot(111)
	ax1.set_xlabel('Number of samples, $N$')
	ax1.set_ylabel('Normalized estimated total-order Sobol index, $\\hat{\\bar{S}}_b^2$, based on $\\mathbb{E}[C]$')
	# b = 0
	t = 0
	if top5:
		rank = 1
	else:
		rank = 4
	for bridge in important_bridges:
		b = bridge_index_lookup[bridge]
		ax1.errorbar(M, mean_S_hat[:,b], xerr=None, yerr=CI_S_hat[:,b], marker='o', ls='none',
					color = colors[t], label='rank '+str(rank)) #label= 'Mean and 95% CI of $\\hat{\\bar{S}}^2$ for bridge '+ bridge
		ax1.axhline(sobol_index_dict_cost[bridge], c=colors[t]),
					# label='bridge ' + bridge) # + ' with $n_{samples} = 185$')  # plot the 'true' value of the Sobol index for this bridge
		rank+=1
		t+= 1

	ax1.set_xlim([0, max(M) + 10])
	ax1.tick_params(axis='both', which='major', labelsize='small')
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.yaxis.set_ticks_position('left')
	ax1.xaxis.set_ticks_position('bottom')
	# get handles
	handles, labels = ax1.get_legend_handles_labels()
	# remove the errorbars
	handles = [h[0] for h in handles]
	# use them in the legend
	ax1.legend(handles, labels, loc='right', bbox_to_anchor = (0.9, 0.45),numpoints=1, frameon=False, prop={'size':12})

	if top5:
		plt.savefig(figures_folder + 'sf_fullr_convergence_top5.png', bbox_inches='tight')
	else:
		plt.savefig(figures_folder + 'sf_fullr_convergence_middle3.png', bbox_inches='tight')


plot_convergence_important_bridges(top5=False)