from __future__ import division
import pickle, copy
import numpy as np
import compute_bridge_sobol_sf_full as cbs
import mahmodel_road_only as mahmodel
import bd_test as bd
import plotly
import plotly.graph_objs as go
import zipfile, shutil
import util
from math import log

mapbox_access_token = 'pk.eyJ1IjoiZ2plZSIsImEiOiJjangzY2F5MDcwMGlpNDhwbWtzbTJ6azBmIn0.P8vS2x_gtfBpWJwWgC3Sbw'

alpha = 48 # dollars per hour
beta = 78*8 # dollars per hour times hours

def update_cs_dict_with_omega(cs_number):

	master_dict = get_master_dict()
	bridge_dict = get_bridge_dict(cs_number=cs_number)

	for b in bridge_dict:
		bridge_dict[b]['omega'] = master_dict[b]['omega']

	filepath = 'input/cs' + str(cs_number) + '_dict.pkl'

	with open(filepath, 'wb') as f:
		pickle.dump(bridge_dict, f)

def get_master_dict():

	with open('input/20140114_master_bridge_dict.pkl', 'rb') as f: # this version includes area for computation of repair cost and bridge-specific retrofit factor omega
		master_dict = pickle.load(f)  # has 1743 keys. One per highway bridge. (NOT BART)

	return master_dict

def get_sf_fullr_dict():

	with open('input/sf_fullr_dict.pkl', 'rb') as f:
		sf_dict = pickle.load(f)


	with open('input/sf_fullr_bridge_ids.pkl', 'rb') as f:
		bridge_ids = pickle.load(f)

	return sf_dict, bridge_ids

def load_undamaged_stats():
	G = mahmodel.get_graph()

	demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
							 'input/superdistricts_centroids_dummies.csv')

	no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = cbs.precompute_network_performance()

	undamaged_stats = [no_damage_travel_time, no_damage_vmt, no_damage_trips_made]

	with open('undamaged_stats_sf_full.pkl','wb') as f:
		pickle.dump(undamaged_stats,f)

	with open('undamaged_stats_sf_full.pkl','rb') as f:
		undamaged_stats = pickle.load(f)

	return undamaged_stats

def load_individual_undamaged_stats():

	undamaged_stats = load_undamaged_stats()

	return undamaged_stats[0], undamaged_stats[1], undamaged_stats[2] # tt, vmt, trips made

def check_symmetric(a, rtol=1e-05, atol=1e-08):
	return np.allclose(a, a.T, rtol=rtol, atol=atol)

def sort_by_performance(bridges,performance): # copied from process_oat_results.py

	n_bridges = len(bridges)

	# create an array with bridge IDs and their performance -- will sort this
	# bridges_structured = np.asarray(bridges, dtype=[('bridge', str)]) # bridge IDs
	# performance_structured = np.asarray(performance, dtype=[('performance', float)]) # performance

	perf_list = [(int(bridges[i]), performance[i]) for i in range(0, n_bridges)]

	print perf_list
	dtype = np.dtype('int,float')
	perf_array = np.array(perf_list, dtype)

	print perf_array

	perf_array_sorted = np.sort(perf_array, axis=0, order='f1')  # sort rows by column 1, the performance, in ascending order

	print perf_array_sorted

	bridges_sorted = [tup[0] for tup in perf_array_sorted]
	perf_sorted = [tup[1] for tup in perf_array_sorted]

	return bridges_sorted, perf_sorted

def plot_bridge_sobol_indices(map_name, bridge_performance_dict):

	county = 'sf'
	county_dict, bridge_ids = get_sf_fullr_dict()
	# bridge_ids_of_interest = county_dict.keys()

	# print 'len(county_dict.keys())', len(county_dict.keys())
	# print 'bridge ids of interest', type(bridge_ids_of_interest)
	# print bridge_ids_of_interest

	# load the NBI dictionary
	with open(
			'/Users/gitanjali/Desktop/TransportNetworks/quick_traffic_model_GROUP/bridges_plot/input/bridge_dict_nbi.pkl',
			'rb') as f:
		bridge_dict_nbi = pickle.load(f)

	# information for bridges of interest
	bridges = []
	lat = []
	long = []
	f = []
	age = []
	traffic = []
	performance = []  # used to define color scale
	#
	# # information for bridges that are not of interest
	# other_bridges = []
	# other_lat = []
	# other_long = []
	# other_f = []
	# other_age = []
	# other_traffic = []
	# other_performance = []

	# scale Sobol' indices to get mappable values
	# new_bridge_performance_dict = scale_sobol_indices(bridge_performance_dict)

	i = 0
	for b in county_dict.keys():  # for every bridge (referenced by original ID)
		new_id = str(county_dict[b]['new_id'])  # get its new ID

		# if b in bridge_ids_of_interest:
		# if bridge_performance_dict[b] > 0:
		bridges.append(b)
		lat.append(bridge_dict_nbi[new_id]['lat'])
		long.append(bridge_dict_nbi[new_id]['long'])
		f.append(county_dict[b]['ext_lnSa'])
		age.append(bridge_dict_nbi[new_id]['age'])
		traffic.append(bridge_dict_nbi[new_id]['traffic'])
		performance.append(bridge_performance_dict[b]) # for color
		# else:
		# 	other_bridges.append(b)
		# 	other_lat.append(bridge_dict_nbi[new_id]['lat'])
		# 	other_long.append(bridge_dict_nbi[new_id]['long'])
		# 	other_f.append(county_dict[b]['ext_lnSa'])
		# 	other_age.append(bridge_dict_nbi[new_id]['age'])
		# 	other_traffic.append(bridge_dict_nbi[new_id]['traffic'])
		# 	other_performance.append(new_bridge_performance_dict[b])
		i += 1

	# print 'len(bridges), len(other_bridges)', len(bridges), len(other_bridges)

	# format the hover text for the map -- automatically includes lat/long; also includes bridge ID (original), fragility fxn parameter
	newline = '<br>'
	text = []
	other_text = []

	for i in range(0, len(bridges)):
		text.append('original ID: ' + bridges[i] + newline +
					'f = ' + str(f[i]) + newline +
					'age = ' + str(age[i]) + newline +
					'daily traffic = ' + str(traffic[i]))

	# for i in range(0, len(other_bridges)):
	# 	other_text.append('original ID: ' + other_bridges[i] + newline +
	# 					  'f = ' + str(other_f[i]) + newline +
	# 					  'age = ' + str(other_age[i]) + newline +
	# 					  'daily traffic = ' + str(other_traffic[i]))

	if county == 'napa':
		map_center_lat = 38.2975
		map_center_long = -122.2869
	elif county == 'sf':
		map_center_lat = 37.7749
		map_center_long = -122.4194
	elif county == 'sm':
		map_center_lat = 37.5630
		map_center_long = -122.3255
	elif county == 'ala':
		map_center_lat = 37.7799
		map_center_long = -122.2822

	# # plot two simple traces
	# data = [go.Scattermapbox(lat=other_lat, lon=other_long, mode='markers',
	# 						 marker=go.scattermapbox.Marker(size=9, color='rgb(172, 188, 241)'), text=other_text,
	# 						 name= county+ ' bridges'),
	# 		go.Scattermapbox(lat=lat, lon=long, mode='markers',
	# 						 marker=go.scattermapbox.Marker(size=9, color='rgb(38, 0, 255)'), text=text,
	# 						 name='bridges of interest')]

	#
	#     # plot in one-shot with custom colorscale
	#     data = [go.Scattermapbox(lat=lat, lon=long, mode='markers',
	#                          marker=go.Marker(size=8, color=performance, colorscale= list(c(0, "rgb(255,0,0)"), list(1, "rgb(0,255,0)")),
	#                                           cauto = F, cmin = , cmax = 1,
	#                                           colorbar=dict(title='Net. Perf. w/o Bridge, s',
	#                                                         showticklabels=True,
	#                                                         tickmode='array', exponentformat='power'),
	#                                           showscale=True, symbol='circle'),
	#                          text=text, name=' bridges'), ]

	# if change:
	# 	colorbar_title = 'Change in Net. Perf. w/o Bridge, s'
	# else:
	# 	colorbar_title = 'Net. Perf. w/o Bridge, s'

	colorbar_title = 'Sobol index'

	# plot in one-shot with colorscale
	data = [go.Scattermapbox(lat=lat, lon=long, mode='markers',  # for outline color
							 marker=go.Marker(size=10, color='rgb(255, 0, 0)',
											  symbol='circle'),),
			go.Scattermapbox(lat=lat, lon=long, mode='markers',  # for interior color
							 marker=go.Marker(size=8, color=performance, colorscale='reds',
											  colorbar=dict(title=colorbar_title,
															showticklabels=True,
															tickmode='array', exponentformat='power'),
											  showscale=True, symbol='circle'),
							 text=text, name=' influential bridges')]
			# go.Scattermapbox(lat=other_lat, lon=other_long, mode='markers',
			# 				 marker=go.Marker(size=8, color='blue', symbol='circle'),
			# 				 text=other_text, name=' non-influential bridges')]

	# data = [go.Scattermapbox(lat=lat, lon=long, mode='markers', # for size
	# 						 marker=go.Marker(size=performance, color='red',
	# 										  showscale=True, symbol='circle'),
	# 						 text=text, name=' bridges'), ]

	# if len(other_bridges)>0: # show legend only if there are multiple traces
	# 	layout = go.Layout(autosize=True, hovermode='closest', showlegend=True,
	# 					   mapbox=go.layout.Mapbox(accesstoken=mapbox_access_token, bearing=0,
	# 											   center=go.layout.mapbox.Center(
	# 												   lat=map_center_lat,
	# 												   lon=map_center_long),
	# 											   zoom=11), )  # center at Napa County, zoom in
	# 	fig = go.Figure(data=data, layout=layout)
	# 	fig.update_layout(legend=dict(x=0.8,
	# 								  y=1))
	# else:
	layout = go.Layout(autosize=True, hovermode='closest', showlegend=False,
						   mapbox=go.layout.Mapbox(accesstoken=mapbox_access_token, bearing=0,
												   center=go.layout.mapbox.Center(
													   lat=map_center_lat,
													   lon=map_center_long),
												   zoom=12), )  # center at Napa County, zoom in
	fig = go.Figure(data=data, layout=layout)

	plotly.offline.plot(fig, filename=map_name + '.html')

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

def get_results_from_pickles(results_directory, results_folder_stub, n_batches, max_batch = 20, scenarios=30, cost='total', retrofit=True, p=False, first_order = False, batch_size=10):
	
	S = scenarios
	D = 10

	bridge_dict, bridge_ids = get_sf_fullr_dict()

	n_bridges =  len(bridge_ids)# how many bridges we considered
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
	sa_matrix = [sa_matrix_full[i] for i in
				 map_indices]  # GB: get the ground_motions for just the scenarios we are interested in


	lnsas = []
	magnitudes = []
	for row in sa_matrix:
		lnsas.append([log(float(sa)) for sa in row[4:]])
		magnitudes.append(float(row[2]))


	no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = cbs.precompute_network_performance()

	j = 0
	skipped =0
	for i in range(0,max_batch):

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

			archive = results_directory + 'run_sf_' + str(i) + '.zip'
			target = 'run_sf_' + str(i)
			print 'archive = ', archive
			print 'results_directory = ', results_directory + target
			with zipfile.ZipFile(archive, 'r') as zip_ref:
				if i > 80:
					zip_ref.extractall(path=results_directory+target)
				else:
					zip_ref.extractall(path=results_directory)

			with open(fX_times_output,'rb') as f:
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
			with open(fX_retrofit_cost_output,'rb') as f:
				temp_fX_retrofit_cost = pickle.load(f)

			temp_fX_avg_times = np.zeros((batch_size,))
			temp_fX_avg_vmts = np.zeros((batch_size,))
			temp_fX_avg_trips = np.zeros((batch_size,))
			temp_fX_exp_indirect_cost = np.zeros((batch_size,))
			temp_fX_exp_direct_cost = np.zeros((batch_size,))
			temp_fX_expected_cost = np.zeros((batch_size,))

			for k in range(0,batch_size):
				# print '*** batch = ', i, ' sample = ', k
				average_travel_time, average_vmt, average_trips_made, average_direct_cost, average_delay_cost, average_connectivity_cost, \
				average_indirect_cost = compute_weighted_average_performance(lnsas, map_weights, num_damage_maps=10, travel_times=temp_fX_times[k,:],
													 vmts=temp_fX_vmts[k,:], trips_made=temp_fX_trips[k,:],
													 no_damage_travel_time=no_damage_travel_time, no_damage_vmt=no_damage_vmt,
													 no_damage_trips_made=no_damage_trips_made, direct_costs=temp_fX_direct_costs[k,:])



				temp_fX_avg_times[k] = average_travel_time
				temp_fX_avg_vmts[k] = average_vmt
				temp_fX_avg_trips[k] = average_trips_made
				temp_fX_exp_direct_cost[k] = average_direct_cost
				temp_fX_exp_indirect_cost[k] = average_indirect_cost # hourly
				temp_fX_expected_cost[k] = 24*125*average_indirect_cost + average_direct_cost

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
			f_X_exp_indirect_cost[j * batch_size:(j + 1) * batch_size, ] = 24*125*temp_fX_exp_indirect_cost # TODO: assume 125 days of closure, 24 hours per day
			f_X_exp_direct_cost[j * batch_size:(j + 1) * batch_size, ] = temp_fX_exp_direct_cost
			#f_X_exp_cost[j * batch_size:(j + 1) * batch_size, ] = temp_fX_expected_cost
			f_X_exp_cost[j * batch_size:(j + 1) * batch_size, ] = 24*125*temp_fX_exp_indirect_cost + temp_fX_exp_direct_cost # TODO: assume 125 days of closure, 24 hours per day
			f_X_ret_cost[j * batch_size:(j + 1) * batch_size, ] = temp_fX_retrofit_cost

			with open(fV_times_output,'rb') as f:
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
			with open(fV_retrofit_cost_output,'rb') as f:
				temp_fV_retrofit_cost = pickle.load(f)

			# print 'temp_fV_avg_times.shape, temp_fV_times.shape', temp_fV_avg_times.shape, temp_fV_times.shape
			# print 'temp_fV_direct_costs.shape = ', temp_fV_direct_costs.shape


			temp_fV_avg_times = np.zeros((batch_size,n_bridges))
			temp_fV_avg_vmts = np.zeros((batch_size,n_bridges))
			temp_fV_avg_trips = np.zeros((batch_size,n_bridges))
			temp_fV_exp_indirect_cost = np.zeros((batch_size,n_bridges))
			temp_fV_exp_direct_cost = np.zeros((batch_size,n_bridges))
			temp_fV_expected_cost = np.zeros((batch_size,n_bridges))

			for k in range(0,batch_size):
				for l in range(0,n_bridges):
					# print '*** batch = ', i, ' sample = ', k
					average_travel_time, average_vmt, average_trips_made, average_direct_cost, average_delay_cost, average_connectivity_cost, \
					average_indirect_cost = compute_weighted_average_performance(lnsas, map_weights, num_damage_maps=10, travel_times=temp_fV_times[k,:,l],
														 vmts=temp_fV_vmts[k,:,l], trips_made=temp_fV_trips[k,:,l],
														 no_damage_travel_time=no_damage_travel_time, no_damage_vmt=no_damage_vmt,
														 no_damage_trips_made=no_damage_trips_made, direct_costs=temp_fV_direct_costs[k,:,l]) #TODO -- make sure all slices are correct!!!!

					temp_fV_avg_times[k,l] = average_travel_time
					temp_fV_avg_vmts[k,l] = average_vmt
					temp_fV_avg_trips[k,l] = average_trips_made
					temp_fV_exp_direct_cost[k,l] = average_direct_cost
					temp_fV_exp_indirect_cost[k,l] = average_indirect_cost # hourly
					temp_fV_expected_cost[k,l] = 24*125*average_indirect_cost + average_direct_cost

			assert np.any(temp_fV_exp_indirect_cost == 0) == False, 'Error in correcting fV_exp_indirect_cost.'
			assert np.any(temp_fV_expected_cost == 0) == False, 'Error in correcting fV_expected_cost.'

			f_V_times[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_times
			f_V_trips[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_trips
			f_V_vmts[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_vmts
			f_V_avg_time[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_avg_times
			f_V_avg_trip[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_avg_trips
			f_V_avg_vmt[j * batch_size:(j + 1) * batch_size, :] = temp_fV_avg_vmts
			f_V_delay_costs[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_delay_costs
			f_V_conn_costs[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_conn_costs
			f_V_indirect_costs[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_indirect_costs
			f_V_direct_costs[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_direct_costs
			f_V_exp_indirect_cost[j * batch_size:(j + 1) * batch_size,: ] = 24*125*temp_fV_exp_indirect_cost # TODO: assume 125 days of closure, 24 hours per day
			f_V_exp_direct_cost[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_exp_direct_cost
			# f_V_exp_cost[j * batch_size:(j + 1) * batch_size,: ] = temp_fV_expected_cost
			f_V_exp_cost[j * batch_size:(j + 1) * batch_size, :] = 24 * 125 * temp_fV_exp_indirect_cost + temp_fV_exp_direct_cost # TODO: assume 125 days of closure, 24 hours per day
			f_V_ret_cost[j * batch_size:(j + 1) * batch_size, :] = temp_fV_retrofit_cost

			j += 1

			shutil.rmtree(results_directory + target)

		except:
			print 'skipped f_X and f_V for batch ', i, 'of ', n_batches, folder, directory + folder + 'fX_times' + filename
			skipped += 1

			try:
				shutil.rmtree(results_directory + target)
			except:
				pass

	print 'skipped ', skipped, ' of ', max_batch, ' batches'

	if not first_order: #i.e., if total-order Sobol' indices
		if retrofit:
			print 'here here here'
			# print f_X_exp_direct_cost.shape, f_X_ret_cost.shape
			if cost == 'total':
				temp = f_X_exp_cost + f_X_ret_cost
				print f_X_exp_cost[0], f_X_ret_cost[0], temp[0]
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_cost+f_X_ret_cost, f_V_exp_cost+f_V_ret_cost, normalize=True) # was originally exp_cost - retrofit_cost
			elif cost == 'indirect':
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_indirect_cost+f_X_ret_cost, f_V_exp_indirect_cost+f_V_ret_cost, normalize=True)
			elif cost == 'direct':
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_direct_cost+f_X_ret_cost, f_V_exp_direct_cost+f_V_ret_cost, normalize=True)
		else:
			if cost == 'total':
				print '*** correct setting for Sobol indices based on expected total cost, not including retrofit cost'
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_cost, f_V_exp_cost, normalize=True)
			elif cost == 'indirect':
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_indirect_cost, f_V_exp_indirect_cost, normalize=True)
			elif cost == 'direct':
				S_exp_cost = cbs.compute_total_sobol_precomputed(f_X_exp_direct_cost, f_V_exp_direct_cost, normalize=True)
	else:
		if retrofit:
			# print f_X_exp_direct_cost.shape, f_X_ret_cost.shape
			if cost == 'total':
				print 'right here'
				f_X_exp_cost = f_X_exp_direct_cost + f_X_exp_indirect_cost
				f_V_exp_cost = f_V_exp_direct_cost + f_V_exp_indirect_cost
				print 'sum of f_X_exp_cost in process results ', sum(f_X_exp_cost), sum(f_X_exp_direct_cost), sum(f_X_exp_indirect_cost)
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_cost, f_V_exp_cost, normalize=True)
			# S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_cost-f_X_ret_cost, f_V_exp_cost-f_V_ret_cost, normalize=True)
			elif cost == 'indirect':
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_indirect_cost-f_X_ret_cost, f_V_exp_indirect_cost-f_V_ret_cost, normalize=True)
			elif cost == 'direct':
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_direct_cost-f_X_ret_cost, f_V_exp_direct_cost-f_V_ret_cost, normalize=True)
		else:
			if cost == 'total':
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_cost, f_V_exp_cost, normalize=True)
			elif cost == 'indirect':
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_indirect_cost, f_V_exp_indirect_cost, normalize=True)
			elif cost == 'direct':
				S_exp_cost = cbs.compute_first_order_sobol_precomputed(f_X_exp_direct_cost, f_V_exp_direct_cost, normalize=True)

	print 'for ', cost, ' cost-based Sobol indices, sum = ', sum(S_exp_cost)

	sobol_index_dict = {}
	i = 0
	for b in bridge_ids:
		sobol_index_dict[b] = S_exp_cost[i]
		# print 'b = ', b, ' sobol_index_dict[b] = ', sobol_index_dict[b]
		print b, sobol_index_dict[b]
		i += 1

	#plot_cost_histogram(f_X_indirect_costs)

	# if cost == 'total':
	# 	return sobol_index_dict, f_X_exp_cost, f_V_exp_cost
	# else:
	# 	return sobol_index_dict, f_X_exp_indirect_cost, f_V_exp_indirect_cost, f_X_indirect_costs

	return sobol_index_dict, f_X_exp_cost, f_V_exp_cost, f_X_ret_cost, f_V_ret_cost

def main():

	results_directory = 'sobol_output/run_sf_fullr_total_all/'
	results_folder_stub = 'run_sf_'
	n_batches =  92 # number of batches for which we actually have results
	max_batches = 140 # maximum batch index (or larger)
	n_scenarios = 30 # S= 30 for training
	print '****** sf_fullr results ******'
	sobol_index_dict, _, _, _, _ = get_results_from_pickles(results_directory, results_folder_stub, n_batches, max_batch=max_batches, scenarios=n_scenarios, cost='total', retrofit=False, batch_size=5) # total cost, not including retrofit cost

if __name__ == "__main__":
    # execute only if run as a script
    main()