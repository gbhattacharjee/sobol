from __future__ import division
# from process_results_sf import get_retrofit_results # THIS function is just for total-order Sobol' indices based on expected total cost.
from process_results_sf import load_individual_undamaged_stats, compute_weighted_average_performance
from make_retrofit_samples import import_retrofit_list, make_incremental_retrofit_lists
from bridges_plot import bridges
import pickle
import matplotlib.pyplot as plt
import numpy as np
import util
from math import log

#TODO make sure you are using the correct baseline for all computations
# baseline (expected cost of network performance, not including retrofit cost, when R = 0) on testing set of S = 45
# baseline_testing = 32417786.20037872
# baseline (expected cost of network performance, not including retrofit cost, when R = 0) on testing set of S = 30
# baseline_training = 30992396.47927609

# Parameters that are true for all of the retrofit tests. # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
n_retrofits = 'various'
n_scenarios = 45
dam_maps_per_scenario = 10
filename = '_sf_fullr'

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
	color_dict['Sobol, exp. cost'] = '#CC79A7'
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

def create_individual_retrofit_dict(): #TODO--REVISE to use correct expectation computation

	n_scenarios = 45

	# List of retrofitting each bridge individually.
	retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - individual_rets.csv'
	tests = import_retrofit_list(retrofit_test_filepath)

	# Results of retrofitting bridges one at a time.
	output_folder = 'sobol_output/retrofits/ind/'
	f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, \
	f_X_exp_indirect_cost, f_X_exp_direct_cost, f_X_exp_cost = get_retrofit_results(output_folder, n_scenarios, filename='_sf_fullr')


	results_dict = dict(zip(tests,f_X_exp_cost))

	with open('sobol_output/retrofits/ind_ret_results_dict.pkl','wb') as f:
		pickle.dump(results_dict,f)

def load_individual_retrofit_dict():

	with open('sobol_output/retrofits/ind_ret_results_dict.pkl','rb') as f:
		results_dict = pickle.load(f)

	return results_dict

def sum_individual_effects(ret_list):

	ind_dict = load_individual_retrofit_dict()

	ind_effects = [ind_dict[b] for b in ret_list]

	baseline = 32417786.20037872 # for S = 45, expected total cost not including retrofit cost

	ind_reductions = [ind_effect-baseline for ind_effect in ind_effects]

	result = sum(ind_reductions)

	return result

def get_incremental_retrofit_results_individual(strategy): # INDIVIDUAL EFFECTS -- this sums the individual effect of each bridge retrofit in a retrofit strategy

	if strategy == 'sobol':
		# retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - sobol_rets.csv' # based on incorrect expectation computation
		retrofit_test_filepath = 'sobol_input/sf_fullr_2020_Sobol_results_revised_averages_exp_cost.csv' # based on N = 370 # based on corrected expectation computation
	elif strategy == 'age':
		retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - age_rets.csv'
	elif strategy == 'oat':
		retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - oat_rets.csv'
	elif strategy == 'traffic':
		retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - traffic_rets.csv'
	elif strategy == 'composite':
		retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - composite_rets.csv'
	elif strategy == 'fragility':
		retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - fragility_rets.csv'
	else:
		retrofit_test_filepath = None

	ret_list = import_retrofit_list(retrofit_test_filepath)

	ret_lists = make_incremental_retrofit_lists(ret_list)

	individual_effects = [sum_individual_effects(r) for r in ret_lists]

	return individual_effects

def get_incremental_retrofit_reduction_individual(strategy): # INDIVIDUAL EFFECTS -- computes change relative to R = 0 baseline

	results = get_incremental_retrofit_results_individual(strategy)

	baseline = 32417786.20037872 # for S = 45, expected total cost with REVISED AVERAGE not including retrofit cost

	# compute reduction

	individual_effect_reduction = [r/baseline for r in results]

	# for i in individual_effect_reduction:
	# 	print i

	return individual_effect_reduction

def get_incremental_retrofit_reduction(strategy): # computes the total effect of retrofitting a group of bridges per a retrofitting strategy

	baseline = 32417786.20037872 # for S = 45, expected total cost with REVISED AVERAGE, not including retrofit cost

	print_results = False

	if strategy == 'age':
		# Age-based retrofit strategy.
		output_folder = 'sobol_output/retrofits/age/'
		f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, \
		f_X_exp_indirect_cost, f_X_exp_direct_cost, f_X_exp_cost = get_retrofit_results(output_folder,n_scenarios,filename,print_results)

		# print f_X_indirect_costs.shape, f_X_exp_direct_cost.shape, f_X_exp_cost.shape

		reduction = [(r-baseline)/baseline for r in f_X_exp_cost]
	elif strategy == 'traffic':
		# Age-based retrofit strategy.
		output_folder = 'sobol_output/retrofits/traffic/'
		f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, \
		f_X_exp_indirect_cost, f_X_exp_direct_cost, f_X_exp_cost = get_retrofit_results(output_folder, n_scenarios,
																							   filename,print_results)
		reduction = [(r-baseline)/baseline for r in f_X_exp_cost]
	elif strategy == 'fragility':
		# Age-based retrofit strategy.
		output_folder = 'sobol_output/retrofits/fragility/'
		f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, \
		f_X_exp_indirect_cost, f_X_exp_direct_cost, f_X_exp_cost = get_retrofit_results(output_folder,n_scenarios,
																							   filename,print_results)
		reduction = [(r-baseline)/baseline for r in f_X_exp_cost]
	elif strategy == 'composite':
		# Age-based retrofit strategy.
		output_folder = 'sobol_output/retrofits/composite/'
		f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, \
		f_X_exp_indirect_cost, f_X_exp_direct_cost, f_X_exp_cost = get_retrofit_results(output_folder, n_scenarios,
																							   filename,print_results)
		reduction = [(r-baseline)/baseline for r in f_X_exp_cost]
	elif strategy == 'oat':
		output_folder = 'sobol_output/retrofits/oat/'
		f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, \
		f_X_exp_indirect_cost, f_X_exp_direct_cost, f_X_exp_cost = get_retrofit_results(output_folder,n_scenarios,
																							   filename,print_results)
		reduction = [(r-baseline)/baseline for r in f_X_exp_cost]
	elif strategy == 'sobol':
		# Sobol-index based retrofit strategy
		n_batches = 10
		batch_size = 7
		results = np.zeros(n_batches*batch_size,)
		for i in range(0,n_batches):
			output_folder = 'sobol_output/retrofits/sobol_retrofits/sobol_'+str(i)+'/' # ordering based on corrected expectation computation and N = 370
			# output_folder = 'sobol_output/retrofits/ret_revised_avg/r' + str(i) + '/'
			f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, \
			f_X_exp_indirect_cost, f_X_exp_direct_cost, f_X_exp_cost = get_retrofit_results(output_folder, n_scenarios,
																								   filename,print_results)

			results[i*batch_size:(i+1)*batch_size] = f_X_exp_cost

		reduction = [(r-baseline)/baseline for r in results]
	else:
		reduction = None

	return reduction

def get_retrofit_results(output_folder, n_scenarios, filename='_sf_full', print_results=True):

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


def plot_network_effects():
	# Recreate individual retrofit dict with revised expectation computation.
	# create_individual_retrofit_dict()

	# Parameters that are true for all of the retrofit tests. # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	n_retrofits = 'various'
	n_scenarios = 45
	dam_maps_per_scenario = 10
	filename = '_sf_fullr'

	# # # PRINTING RESULTS TO TRANSFER TO GOOGLE SHEET sf_fullr_2020_network_effects # # # # # # # # # # # # # # # # # # # #
	results_sobol = get_incremental_retrofit_reduction('sobol')
	results_oat = get_incremental_retrofit_reduction('oat')
	results_age = get_incremental_retrofit_reduction('age')
	results_fragility = get_incremental_retrofit_reduction('fragility')
	results_traffic = get_incremental_retrofit_reduction('traffic')
	results_composite = get_incremental_retrofit_reduction('composite')

	results_sobol.append(results_oat[-1]) # neglected to test Sobol' retrofit strategy for R = 71, so copy data from a different strategy (result will be the same since all bridges are retrofitted)

	print 'got results'

	print '*** Results'
	print 'age = ', results_age
	print 'fragility = ', results_fragility
	print 'traffic = ', results_traffic
	print 'composite = ', results_composite
	print 'oat = ', results_oat
	print 'old sobol = ', results_sobol

	print 'len(OAT) = ',  len(results_oat)
	print 'len(net_effect_sobol) = ', len(results_sobol)


	ind_sobol = get_incremental_retrofit_reduction_individual('sobol')
	ind_oat =  get_incremental_retrofit_reduction_individual('oat')
	ind_age =  get_incremental_retrofit_reduction_individual('age')
	ind_traffic = get_incremental_retrofit_reduction_individual('traffic')
	ind_fragility = get_incremental_retrofit_reduction_individual('fragility')
	ind_composite = get_incremental_retrofit_reduction_individual('composite')
	print 'got incremental results'

	print '*** Incremental results'
	print 'age = ', ind_age
	print 'fragility = ', ind_fragility
	print 'traffic = ', ind_traffic
	print 'composite = ', ind_composite
	print 'oat = ', ind_oat
	print 'old sobol = ', ind_sobol

	print 'len(OAT) = ',  len(ind_oat)
	print 'len(net_effect_sobol) = ', len(ind_sobol)

	net_effect_sobol = [-1*(results_sobol[i]-ind_sobol[i])*100 for i in range(0,len(results_sobol))]
	net_effect_oat = [-1*(results_oat[i]-ind_oat[i])*100 for i in range(0,len(results_oat))]
	net_effect_age = [-1*(results_age[i]-ind_age[i])*100 for i in range(0,len(results_age))] # percent difference between retrofitting bridges as a group and the sum of their individual retrofit effects
	net_effect_traffic = [-1*(results_traffic[i]-ind_traffic[i])*100 for i in range(0,len(results_traffic))]
	net_effect_fragility = [-1*(results_fragility[i]-ind_fragility[i])*100 for i in range(0,len(results_fragility))]
	net_effect_composite = [-1*(results_composite[i]-ind_composite[i])*100 for i in range(0,len(results_composite))]

	print '*** Net effects'
	print 'age = ', net_effect_age
	print 'fragility = ', net_effect_fragility
	print 'traffic = ', net_effect_traffic
	print 'composite = ', net_effect_composite
	print 'oat = ', net_effect_oat
	print 'old sobol = ', net_effect_sobol

	print 'len(OAT) = ',  len(net_effect_oat)
	print 'len(net_effect_sobol) = ', len(net_effect_sobol)

	marker_style = 'o'
	fig = plt.figure()
	ax = fig.add_subplot(111) # colors match those in the retrofit results plot ('sf_fullr_cost_vs_nbridges_positive.png')
	ax.plot(np.arange(0,len(net_effect_age),1), net_effect_age, marker=marker_style, label='age', color=get_color('age'))
	ax.plot(np.arange(0,len(net_effect_fragility),1), net_effect_fragility, marker=marker_style, label='fragility', color=get_color('fragility'))
	ax.plot(np.arange(0,len(net_effect_traffic),1), net_effect_traffic, marker=marker_style, label='traffic volume', color=get_color('traffic'))
	ax.plot(np.arange(0,len(net_effect_composite),1), net_effect_composite, marker=marker_style, label='composite', color=get_color('composite'))
	ax.plot(np.arange(0,len(net_effect_oat),1), net_effect_oat, marker=marker_style, label = 'OAT', color=get_color('OAT'))
	ax.plot(np.arange(0,len(net_effect_sobol),1), net_effect_sobol, marker=marker_style, label='Sobol, $\\mathbb{E}[C]$', color=get_color('Sobol'))
	ax.set_xlim([0, 72])
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	handles, labels = ax.get_legend_handles_labels()

	handles = [handles[5], handles[4], handles[0], handles[1], handles[3],  handles[2]]
	labels = [labels[5], labels[4], labels[0], labels[1], labels[3], labels[2]]

	plt.legend(handles, labels, loc='best', frameon=False)
	plt.xlabel('Number of retrofitted bridges, $R$')
	plt.ylabel('Network effect')
	plt.savefig('figs/network_effects_all_strategies.png', bbox_inches='tight')

plot_network_effects()