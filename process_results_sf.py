from __future__ import division
import pickle, copy
import numpy as np
import compute_bridge_sobol_sf_full as cbs
from itertools import permutations
import mahmodel_road_only as mahmodel
import bd_test as bd
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import matplotlib.ticker as mticker
import bridges_plot.bridges as bplot
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

def run_simple_oat(cs_number):

	partial_dict = get_bridge_dict(cs_number)
	bridge_dict = get_bridge_dict(cs_number)

	bridges = get_bridge_ids(cs_number)

	oat_dict = {}
	for bridge in bridges:
		oat_dict[bridge] = {}

		demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
								 'input/superdistricts_centroids_dummies.csv')

		no_damage_travel_time, no_damage_vmt, no_damage_trips_made = load_individual_undamaged_stats()


		index, road_bridges_out, travel_time, vmt, trips_made, damaged_bridges_internal = mahmodel.compute_road_performance(None,[bridge], demand, no_damage_travel_time, no_damage_vmt, no_damage_trips_made, partial_dict, 0)
		assert road_bridges_out == 1, 'Error: more bridges damaged than the one intended.'


		oat_dict[bridge]['tt'] = travel_time
		oat_dict[bridge]['vmt'] = vmt
		oat_dict[bridge]['trips'] = trips_made
		oat_dict[bridge]['exp_ind_cost'] = alpha*max(0, (travel_time-no_damage_travel_time)/3600) + beta*max(0, no_damage_trips_made-trips_made) # dollars per hour
		oat_dict[bridge]['exp_direct_cost'] = bridge_dict[str(bridge)]['area']*293*0.25 # dollars
		oat_dict[bridge]['retrofit_cost'] = 0.3*(bridge_dict[str(bridge)]['area']*293*0.25) # dollars

		print 'done with bridge ', bridge

	output_dict_name = 'oat_dict_cs' + str(cs_number) + '.pkl'
	with open(output_dict_name,'wb') as f:
		pickle.dump(oat_dict,f)

def print_simple_oat_results(cs_number):

	output_dict_name = 'oat_dict_cs' + str(cs_number) + '.pkl'
	with open(output_dict_name,'rb') as f:
		oat_dict = pickle.load(f)

	for bridge in oat_dict.keys():
		print bridge, oat_dict[bridge]['exp_direct_cost'], oat_dict[bridge]['exp_ind_cost'] # note these are actually not expected costs, just costs -- error in naming!

def plot_simple_oat_results(cs_number):


	no_damage_travel_time, no_damage_vmt, no_damage_trips_made = load_individual_undamaged_stats()

	output_dict_name = 'oat_dict_cs' + str(cs_number) + '.pkl'
	with open(output_dict_name,'rb') as f:
		oat_dict = pickle.load(f)

	bridges = oat_dict.keys()
	indirect_cost = [oat_dict[bridge]['exp_ind_cost'] for bridge in bridges]

	bridges_sorted, indirect_cost_sorted = sort_by_performance(bridges, indirect_cost)

	delay_sorted = [max(0, oat_dict[str(b)]['tt']-no_damage_travel_time)/3600 for b in bridges_sorted]
	trips_lost_sorted = [max(0, no_damage_trips_made-oat_dict[str(b)]['trips']) for b in bridges_sorted]

	print trips_lost_sorted

	delay_cost_sorted = [alpha*max(0, oat_dict[str(b)]['tt']-no_damage_travel_time) for b in bridges_sorted]
	conn_cost_sorted = [beta*max(0, no_damage_trips_made-oat_dict[str(b)]['trips']) for b in bridges_sorted]

	B = len(bridges)
	if B > 20:
		plot_limit = 20
	else:
		plot_limit = B

	fig_folder = 'figs/'
	county = 'cs'+str(cs_number)

	# Stacked bar chart for total one-day costs for Napa County split up into components -- costs due to delays and costs due to lost trips (connectivity loss).
	y_position = np.arange(0, B, 1)
	fig = plt.figure()  # This is a plot of the simple OAT analysis results (i.e. not using scenarios, just using baseline performance computation).
	ax = fig.add_subplot(111)
	ax.set_xlabel('Hourly cost, 2020 USD')
	ax.set_ylabel('ID of Damaged Bridge')
	ax.barh(y_position[0:plot_limit], delay_cost_sorted[B - plot_limit:B], align='center', alpha=1,
			label='cost due to delays', color='#f3c7c4')
	ax.barh(y_position[0:plot_limit], conn_cost_sorted[B - plot_limit:B], align='center', alpha=1,
			label='cost due to lost trips',
			left=np.asarray(delay_cost_sorted[B - plot_limit:B]), color='#872435')
	ax.tick_params(axis='y', which='major', labelsize=10)
	ax.set_yticks(y_position[0:plot_limit])
	ax.set_yticklabels(bridges_sorted[B - plot_limit:B])
	ax.legend(loc='best', prop={'size': 10})
	plt.savefig(fig_folder + county + '_OAT_indirect_cost_components', bbox_inches='tight')

	# Stacked bar chart for total one-day costs for Napa County split up into components -- costs due to delays and costs due to lost trips (connectivity loss).
	y_position = np.arange(0, B, 1)
	fig = plt.figure()  # This is a plot of the simple OAT analysis results (i.e. not using scenarios, just using baseline performance computation).
	ax = fig.add_subplot(111)
	ax.set_xlabel('Cost, 2020 USD')
	ax.set_ylabel('ID of Damaged Bridge')
	ax.barh(y_position[0:plot_limit], delay_sorted[B - plot_limit:B], align='center', alpha=1,
			label='delays', color='#f3c7c4')
	ax.tick_params(axis='y', which='major', labelsize=10)
	ax.set_yticks(y_position[0:plot_limit])
	ax.set_yticklabels(bridges_sorted[B - plot_limit:B])
	ax.legend(loc='best', prop={'size': 10})
	plt.savefig(fig_folder + county + '_OAT_delays', bbox_inches='tight')

def plot_simple_oat_results_with_network_effect(cs_number):
	# G = mahmodel.get_graph()
	#
	# demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
	# 						 'input/superdistricts_centroids_dummies.csv')
	#
	# no_damage_travel_time, no_damage_vmt, no_damage_trips_made, _ = cbs.precompute_network_performance(G, demand)

	undamaged_stats = load_undamaged_stats()
	no_damage_travel_time = undamaged_stats[0]
	no_damage_vmt = undamaged_stats[1]
	no_damage_trips_made = undamaged_stats[2]

	output_dict_name = 'oat_dict_cs' + str(cs_number) + '.pkl'
	with open(output_dict_name, 'rb') as f:
		oat_dict = pickle.load(f)

	bridges = oat_dict.keys()
	indirect_cost = [oat_dict[bridge]['exp_ind_cost'] for bridge in bridges]

	# add in selected cases to compare network effects
	damaged_bridges = [['1081', '951'], ['1081', '951', '976'], ['976', '1081']]

	i = 0
	for damaged_bridge_set in damaged_bridges:
		network_bridges_temp_id = str(i)
		network_bridges, network_tt, network_trips, network_indirect_cost = test_network_effect(damaged_bridge_set)
		bridges.append(network_bridges_temp_id)
		indirect_cost.append(network_indirect_cost)

		oat_dict[network_bridges_temp_id] = {}
		oat_dict[network_bridges_temp_id]['tt'] = network_tt
		oat_dict[network_bridges_temp_id]['trips'] = network_trips
		oat_dict[network_bridges_temp_id]['exp_ind_cost'] = network_indirect_cost
		i += 1

	max_i = i -1

	bridges_sorted, indirect_cost_sorted = sort_by_performance(bridges, indirect_cost)


	delay_sorted = [max(0, oat_dict[str(b)]['tt'] - no_damage_travel_time) / 3600 for b in bridges_sorted]
	trips_lost_sorted = [max(0, no_damage_trips_made - oat_dict[str(b)]['trips']) for b in bridges_sorted]


	delay_cost_sorted = [alpha * max(0, oat_dict[str(b)]['tt'] - no_damage_travel_time) for b in bridges_sorted]
	conn_cost_sorted = [78 * 8*max(0, no_damage_trips_made - oat_dict[str(b)]['trips']) for b in bridges_sorted]

	B = len(bridges)
	if B > 50:
		plot_limit = 50
	else:
		plot_limit = B

	fig_folder = 'figs/'
	county = 'sft2'

	# Stacked bar chart for total one-day costs for Napa County split up into components -- costs due to delays and costs due to lost trips (connectivity loss).
	y_position = np.arange(0, B, 1)
	print bridges_sorted
	j = 0
	for bridge in bridges_sorted:
		if bridge > max_i:
			bridges_sorted[j] = str(bridge)
		elif bridge == 0:
			bridges_sorted[j] = str('1081 and 951')
		elif str(bridge) == str(1):
			bridges_sorted[j] = str('1081, 951, and 976')
		elif str(bridge) == str(2):
			bridges_sorted[j] = str('1081 and 976')
		else:
			pass

		j += 1

	# bridges_sorted = [str(bridge) if bridge > max_i else '1081 and 951' for bridge in bridges_sorted]
	print bridges_sorted
	print bridges_sorted[B - plot_limit:B]

	fig = plt.figure()  # This is a plot of the simple OAT analysis results (i.e. not using scenarios, just using baseline performance computation).
	ax = fig.add_subplot(111)
	ax.set_xlabel('Hourly cost, 2020 USD')
	ax.set_ylabel('ID of Damaged Bridge')
	ax.barh(y_position[0:plot_limit], delay_cost_sorted[B - plot_limit:B], align='center', alpha=1,
			label='cost due to delays', color='#f3c7c4')
	ax.barh(y_position[0:plot_limit], conn_cost_sorted[B - plot_limit:B], align='center', alpha=1,
			label='cost due to lost trips',
			left=np.asarray(delay_cost_sorted[B - plot_limit:B]), color='#872435')
	ax.tick_params(axis='y', which='major', labelsize=10)
	ax.set_yticks(y_position[0:plot_limit])
	ax.set_yticklabels(bridges_sorted[B - plot_limit:B])
	ax.legend(loc='best', prop={'size': 10})
	plt.savefig(fig_folder + county + '_OAT_indirect_cost_components', bbox_inches='tight')

	# Stacked bar chart for total one-day costs for Napa County split up into components -- costs due to delays and costs due to lost trips (connectivity loss).
	y_position = np.arange(0, B, 1)
	fig = plt.figure()  # This is a plot of the simple OAT analysis results (i.e. not using scenarios, just using baseline performance computation).
	ax = fig.add_subplot(111)
	ax.set_xlabel('Hours')
	ax.set_ylabel('ID of Damaged Bridge')
	ax.barh(y_position[0:plot_limit], delay_sorted[B - plot_limit:B], align='center', alpha=1,
			label='delays', color='#f3c7c4')
	ax.tick_params(axis='y', which='major', labelsize=10)
	ax.set_yticks(y_position[0:plot_limit])
	ax.set_yticklabels(bridges_sorted[B - plot_limit:B])
	ax.legend(loc='best', prop={'size': 10})
	plt.savefig(fig_folder + county + '_OAT_delays', bbox_inches='tight')

def plot_simple_oat_results_total_cost(cs_number):



	fig_folder = 'figs/'
	county = 'cs'+str(cs_number)

	no_damage_travel_time, no_damage_vmt, no_damage_trips_made = load_individual_undamaged_stats()

	output_dict_name = 'oat_dict_cs' + str(cs_number) + '.pkl'
	with open(output_dict_name, 'rb') as f:
		oat_dict = pickle.load(f)

	bridge_dict = get_bridge_dict(cs_number)

	bridges = get_bridge_ids(cs_number)
	B = len(bridges)
	if B > 20:
		plot_limit = 20
	else:
		plot_limit = B


	indirect_cost = [oat_dict[bridge]['exp_ind_cost']*24*125 for bridge in bridges]
	delay_cost = [alpha * max(0, (oat_dict[str(b)]['tt'] - no_damage_travel_time))/3600*24*125 for b in bridges]
	conn_cost = [beta * max(0, no_damage_trips_made - oat_dict[str(b)]['trips'])*24*125 for b in bridges]
	indirect_cost_check = np.sum((np.asarray(delay_cost), np.asarray(conn_cost)),axis=0)

	direct_cost = [bridge_dict[bridge]['area']*293 for bridge in bridges]
	total_cost = [indirect_cost[b] + direct_cost[b] for b in range(0,len(indirect_cost))]

	# SORT BRIDGES BY TOTAL COST AND PLOT
	bridges_sorted, total_cost_sorted = sort_by_performance(bridges, total_cost)
	delay_cost_sorted = [alpha * max(0, (oat_dict[str(b)]['tt'] - no_damage_travel_time))/3600*24*125 for b in bridges_sorted]
	conn_cost_sorted = [beta * max(0, no_damage_trips_made - oat_dict[str(b)]['trips'])*24*125 for b in bridges_sorted]
	direct_cost_sorted = [bridge_dict[str(bridge)]['area']*293 for bridge in bridges_sorted]

	direct_cost_position = np.sum((np.asarray(delay_cost_sorted[B-plot_limit:B]), np.asarray(conn_cost_sorted[B - plot_limit:B])),axis=0)

	# Stacked bar chart for total one-day costs for Napa County split up into components -- costs due to delays and costs due to lost trips (connectivity loss).
	y_position = np.arange(0, B, 1)
	fig = plt.figure()  # This is a plot of the simple OAT analysis results (i.e. not using scenarios, just using baseline performance computation).
	ax = fig.add_subplot(111)
	ax.set_xlabel('Cost, 2020 USD')
	ax.set_ylabel('ID of Damaged Bridge')
	ax.barh(y_position[0:plot_limit], delay_cost_sorted[B - plot_limit:B], align='center', alpha=1,
			label='cost due to delays', color='#f3c7c4')
	ax.barh(y_position[0:plot_limit], conn_cost_sorted[B - plot_limit:B], align='center', alpha=1,
			label='cost due to lost trips',
			left=np.asarray(delay_cost_sorted[B - plot_limit:B]), color='#872435')
	ax.barh(y_position[0:plot_limit], direct_cost_sorted[B - plot_limit:B], align='center', alpha=1,
			label='cost due to repairs',
			left=direct_cost_position, color='red')
	ax.tick_params(axis='y', which='major', labelsize=6)
	ax.set_yticks(y_position[0:plot_limit])
	ax.set_yticklabels(bridges_sorted[B - plot_limit:B])
	ax.set_xscale('log')
	ax.legend(loc='best', prop={'size': 10})
	ax.set_title('Total cost of bridge damage per OAT analysis')
	plt.savefig(fig_folder + county + '_OAT_total_cost_components', bbox_inches='tight')

	# SORT BRIDGES BY INDIRECT COST AND PLOT
	bridges_sorted, indirect_cost_sorted = sort_by_performance(bridges, indirect_cost)
	delay_cost_sorted = [alpha * max(0, (oat_dict[str(b)]['tt'] - no_damage_travel_time)) / 3600 * 24 * 125 for b in
						 bridges_sorted]
	conn_cost_sorted = [beta * max(0, no_damage_trips_made - oat_dict[str(b)]['trips']) * 24 * 125 for b in
						bridges_sorted]

	y_position = np.arange(0, B, 1)
	fig = plt.figure()  # This is a plot of the simple OAT analysis results (i.e. not using scenarios, just using baseline performance computation).
	ax = fig.add_subplot(111)
	ax.set_xlabel('Cost, 2020 USD')
	ax.set_ylabel('ID of Damaged Bridge')
	ax.barh(y_position[0:plot_limit], delay_cost_sorted[B - plot_limit:B], align='center', alpha=1,
			label='cost due to delays', color='#f3c7c4')
	ax.barh(y_position[0:plot_limit], conn_cost_sorted[B - plot_limit:B], align='center', alpha=1,
			label='cost due to lost trips',
			left=np.asarray(delay_cost_sorted[B - plot_limit:B]), color='#872435')
	ax.tick_params(axis='y', which='major', labelsize=6)
	ax.set_yticks(y_position[0:plot_limit])
	ax.set_yticklabels(bridges_sorted[B - plot_limit:B])
	ax.set_xscale('log')
	ax.legend(loc='best', prop={'size': 10})
	ax.set_title('Indirect cost of bridge damage per OAT analysis')
	plt.savefig(fig_folder + county + '_OAT_indirect_cost_components', bbox_inches='tight')

def scale_sobol_indices(sobol_index_dict):

	values = sobol_index_dict.values()

	min = np.min(values)
	max = np.max(values)

	old_range = max - min

	new_min = 0
	new_max = 1

	new_range = new_max - new_min

	new_dict = copy.deepcopy(sobol_index_dict)

	for k in sobol_index_dict.keys():
		temp = sobol_index_dict[k]
		new_value = (((temp - min) * new_range) / old_range) + new_min
		new_dict[k] = new_value

	return new_dict

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

def test_network_effect(damaged_bridges):

	partial_dict = get_dict('sft2_dict')

	# if B == 2:
	# 	damaged_bridges = ['1081', '951']
	# elif B == 3:
	# 	damaged_bridges = ['1081', '951', '976']
	# elif B == 22:
	# 	damaged_bridges = ['976', '1081']

	undamaged_stats = load_undamaged_stats()
	no_damage_travel_time = undamaged_stats[0]
	no_damage_vmt = undamaged_stats[1]
	no_damage_trips_made = undamaged_stats[2]

	demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
							 'input/superdistricts_centroids_dummies.csv')

	index, road_bridges_out, travel_time, vmt, trips_made, damaged_bridges_internal = mahmodel.compute_road_performance(
		None, damaged_bridges, demand, no_damage_travel_time, no_damage_vmt, no_damage_trips_made, partial_dict, 0)

	indirect_cost = alpha*max(travel_time-no_damage_travel_time, 0) + beta*max(no_damage_trips_made - trips_made, 0)

	return damaged_bridges, travel_time, trips_made, indirect_cost

def make_retrofit_samples_sf(cs_number, n_retrofits, retrofit_lists, retrofit_sample_filepath,custom_filename=None):
	# Make samples of fragility function parameters for different sets of retrofits based on age, weakness, traffic volume,
	# and total-order Sobol index as computed with N = 90, S = 10, D = 3 for the B = 18 bridges of sf_testbed_new
	# in San Francisco County.

	n_retrofit_strategies = len(retrofit_lists)
	partial_dict = get_bridge_dict(cs_number)
	bridge_ids = get_bridge_ids(cs_number)
	n_bridges = len(bridge_ids)

	# STEP 5c. Generate the lists of fragility parameters for each retrofitting strategy test. Each set of fragility
	# parameters will later be passed as the input 'x' to mahmodel_road_only_napa_2.py.

	retrofit_x = np.zeros((n_retrofit_strategies, n_bridges)) # fragility parameters for bridges retrofitted per each retrofit strategy

	r_counts = np.zeros((n_retrofit_strategies,))
	if n_retrofits > 0:
		i = 0 # counter for bridge index
		for b in bridge_ids:  # for the bridges in the case study
			ff_og = partial_dict[str(b)]['ext_lnSa']  # original fragility function parameter
			ff_ret = partial_dict[str(b)]['ext_lnSa'] * partial_dict[str(b)]['omega']  # retrofitted fragility function parameter
			for r in range(0,n_retrofit_strategies): # for each retrofit strategy
				print b, retrofit_lists[r][0:n_retrofits], b in retrofit_lists[r][0:n_retrofits]
				if b in retrofit_lists[r][0:n_retrofits]:
					# print b in retrofit_lists[r][0:n_retrofits]
					retrofit_x[r, i] = ff_ret
					r_counts[r] += 1
				else:
					retrofit_x[r, i] = ff_og
			i += 1
	elif n_retrofits == 0:
		retrofit_x = np.zeros((1,n_bridges))
		for i in range(0,n_bridges):
			retrofit_x[0,i] = partial_dict[str(bridge_ids[i])]['ext_lnSa']
	else:
		'Number of retrofits is not valid.'

	print r_counts

	assert np.sum(r_counts) == n_retrofit_strategies*n_retrofits, 'The number of retrofits specified by the user was not carried out.'

	# Store the fragility function parameter samples in a single array and save as a pickle.
	retrofit_samples = np.asarray(retrofit_x)

	print 'retrofit samples'
	print retrofit_samples

	if custom_filename is not None:
		retrofit_sample_file = custom_filename
	else:
		retrofit_sample_file = 'retrofit_samples_cs' + str(cs_number) + '_r_' + str(n_retrofits) + '.pkl'

	retrofit_sample_filepath = retrofit_sample_filepath +retrofit_sample_file

	with open(retrofit_sample_filepath, 'wb') as f:
		pickle.dump(retrofit_samples, f)

	print 'stored at: ', retrofit_sample_filepath

	print 'Done generating samples for retrofit strategies with R = ', n_retrofits, '.'

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

def get_raw_results_from_pickles(results_directory, results_folder_stub, n_batches, max_batch=20, scenarios=30,
								 cost='total', retrofit=False, p=False, first_order=False, batch_size=10):
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

	# Adding info to compute expectations correctly from raw damage map data.
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
		print results_directory + folder

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
			if i < 80: #TODO -- fix this -- not sure why I'm getting different behavior for same function.
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
			with open(fX_delay_costs_output, 'rb') as f:
				temp_fX_delay_costs = pickle.load(f)
			with open(fX_conn_costs_output, 'rb') as f:
				temp_fX_conn_costs = pickle.load(f)
			with open(fX_indirect_costs_output, 'rb') as f:
				temp_fX_indirect_costs = pickle.load(f)
			with open(fX_direct_costs_output, 'rb') as f:
				temp_fX_direct_costs = pickle.load(f)
			# with open(fX_exp_indirect_cost_output, 'rb') as f:
			# 	temp_fX_exp_indirect_cost = pickle.load(f)
			# with open(fX_exp_direct_cost_output, 'rb') as f:
			# 	temp_fX_exp_direct_cost = pickle.load(f)
			# with open(fX_expected_cost_output, 'rb') as f:
			# 	temp_fX_expected_cost = pickle.load(f)
			with open(fX_retrofit_cost_output, 'rb') as f:
				temp_fX_retrofit_cost = pickle.load(f)

			# Compute expectations correctly from raw damage map data.
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
																			 direct_costs=temp_fX_direct_costs[k,:])

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

			f_X_delay_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_delay_costs
			f_X_conn_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_conn_costs
			f_X_indirect_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_indirect_costs
			f_X_direct_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_direct_costs
			f_X_exp_indirect_cost[j * batch_size:(j + 1) * batch_size, ] = 24 * 125 * temp_fX_exp_indirect_cost  # TODO: assume 125 days of closure, 24 hours per day
			f_X_exp_direct_cost[j * batch_size:(j + 1) * batch_size, ] = temp_fX_exp_direct_cost
			f_X_exp_cost[j * batch_size:(j + 1) * batch_size, ] = 24 * 125 * temp_fX_exp_indirect_cost + temp_fX_exp_direct_cost  # TODO: assume 125 days of closure, 24 hours per day
			f_X_ret_cost[j * batch_size:(j + 1) * batch_size, ] = temp_fX_retrofit_cost

			with open(fV_times_output, 'rb') as f:
				temp_fV_times = pickle.load(f)
			with open(fV_trips_output, 'rb') as f:
				temp_fV_trips = pickle.load(f)
			with open(fV_vmts_output, 'rb') as f:
				temp_fV_vmts = pickle.load(f)
			# with open(fV_avg_times_output, 'rb') as f:
			# 	temp_fV_avg_times = pickle.load(f)
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
			# with open(fV_expected_cost_output, 'rb') as f:
			# 	temp_fV_expected_cost = pickle.load(f)
			with open(fV_retrofit_cost_output, 'rb') as f:
				temp_fV_retrofit_cost = pickle.load(f)

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
																				 direct_costs=temp_fV_direct_costs[k,:])

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

			j += 1 # tracks the actual number of batches that we get

			shutil.rmtree(results_directory + 'run_sf_' + str(i))

		except:
			print 'skipped f_X and f_V for batch ', i, 'of ', n_batches, folder, directory + folder + 'fX_times' + filename
			skipped += 1

			try:
				shutil.rmtree(extracted_file)
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

	# return f_X_indirect_costs, f_X_direct_costs, f_V_indirect_costs, f_V_direct_costs, f_X_exp_cost, f_V_exp_cost

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

			extracted_file = results_directory + 'run_sf_' + str(i) + '.zip'
			with zipfile.ZipFile(extracted_file, 'r') as zip_ref:
				zip_ref.extractall(results_directory)


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

			shutil.rmtree(results_directory + 'run_sf_' + str(i))

		except:
			print 'skipped f_X and f_V for batch ', i, 'of ', n_batches, folder, directory + folder + 'fX_times' + filename
			skipped += 1

			try:
				shutil.rmtree(extracted_file)
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


def get_results_from_pickles_ORIGINAL(results_directory, results_folder_stub, n_batches, max_batch=20, scenarios=30,
							 cost='total', retrofit=True, p=False, first_order=False, batch_size=10):
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
			with zipfile.ZipFile(extracted_file, 'r') as zip_ref:
				zip_ref.extractall(results_directory)

			with open(fX_times_output, 'rb') as f:
				temp_fX_times = pickle.load(f)
			with open(fX_trips_output, 'rb') as f:
				temp_fX_trips = pickle.load(f)
			with open(fX_vmts_output, 'rb') as f:
				temp_fX_vmts = pickle.load(f)
			with open(fX_avg_times_output, 'rb') as f:
				temp_fX_avg_times = pickle.load(f)
			with open(fX_avg_trips_output, 'rb') as f:
				temp_fX_avg_trips = pickle.load(f)
			with open(fX_avg_vmts_output, 'rb') as f:
				temp_fX_avg_vmts = pickle.load(f)
			with open(fX_delay_costs_output, 'rb') as f:
				temp_fX_delay_costs = pickle.load(f)
			with open(fX_conn_costs_output, 'rb') as f:
				temp_fX_conn_costs = pickle.load(f)
			with open(fX_indirect_costs_output, 'rb') as f:
				temp_fX_indirect_costs = pickle.load(f)
			with open(fX_direct_costs_output, 'rb') as f:
				temp_fX_direct_costs = pickle.load(f)
			with open(fX_exp_indirect_cost_output, 'rb') as f:
				temp_fX_exp_indirect_cost = pickle.load(f)
			with open(fX_exp_direct_cost_output, 'rb') as f:
				temp_fX_exp_direct_cost = pickle.load(f)
			with open(fX_expected_cost_output, 'rb') as f:
				temp_fX_expected_cost = pickle.load(f)
			with open(fX_retrofit_cost_output, 'rb') as f:
				temp_fX_retrofit_cost = pickle.load(f)

			# print type(f_X_times), len(f_X_times)

			f_X_times[j * batch_size:(j + 1) * batch_size, ] = temp_fX_times
			f_X_trips[j * batch_size:(j + 1) * batch_size, ] = temp_fX_trips
			f_X_vmts[j * batch_size:(j + 1) * batch_size, ] = temp_fX_vmts
			f_X_avg_time[j * batch_size:(j + 1) * batch_size, ] = temp_fX_avg_times
			f_X_avg_trip[j * batch_size:(j + 1) * batch_size, ] = temp_fX_avg_trips
			f_X_avg_vmt[j * batch_size:(j + 1) * batch_size, ] = temp_fX_avg_vmts
			f_X_delay_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_delay_costs
			f_X_conn_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_conn_costs
			f_X_indirect_costs[j * batch_size:(j + 1) * batch_size, ] = temp_fX_indirect_costs
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
			with open(fV_avg_trips_output, 'rb') as f:
				temp_fV_avg_trips = pickle.load(f)
			with open(fV_avg_vmts_output, 'rb') as f:
				temp_fV_avg_vmts = pickle.load(f)
			with open(fV_delay_costs_output, 'rb') as f:
				temp_fV_delay_costs = pickle.load(f)
			with open(fV_conn_costs_output, 'rb') as f:
				temp_fV_conn_costs = pickle.load(f)
			with open(fV_indirect_costs_output, 'rb') as f:
				temp_fV_indirect_costs = pickle.load(f)
			with open(fV_direct_costs_output, 'rb') as f:
				temp_fV_direct_costs = pickle.load(f)
			with open(fV_exp_indirect_cost_output, 'rb') as f:
				temp_fV_exp_indirect_cost = pickle.load(f)
			with open(fV_exp_direct_cost_output, 'rb') as f:
				temp_fV_exp_direct_cost = pickle.load(f)
			with open(fV_expected_cost_output,'rb') as f:
				temp_fV_expected_cost = pickle.load(f)
			with open(fV_retrofit_cost_output,'rb') as f:
				temp_fV_retrofit_cost = pickle.load(f)

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

			shutil.rmtree(results_directory + 'run_sf_' + str(i))

		except:
			print 'skipped f_X and f_V for batch ', i, 'of ', n_batches, folder, directory + folder + 'fX_times' + filename
			skipped += 1

			try:
				shutil.rmtree(extracted_file)
			except:
				pass

	print 'skipped ', skipped, ' of ', max_batch, ' batches'

	if not first_order:
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
		print 'b = ', b, ' sobol_index_dict[b] = ', sobol_index_dict[b]
		i += 1

	#plot_cost_histogram(f_X_indirect_costs)

	# if cost == 'total':
	# 	return sobol_index_dict, f_X_exp_cost, f_V_exp_cost
	# else:
	# 	return sobol_index_dict, f_X_exp_indirect_cost, f_V_exp_indirect_cost, f_X_indirect_costs

	return sobol_index_dict, f_X_exp_cost, f_V_exp_cost, f_X_ret_cost, f_V_ret_cost

def plot_sobol_index_comparison(s1 = 20, s2 = 39):


	results_directory = '/Users/gitanjali/Desktop/TransportNetworks/quick_traffic_model_GROUP/sobol_output/results_sft2/'
	results_folder_stub = 'run_sft2_'
	file_suffix = '_sft2.pkl'  # suffix

	s1_dict = get_results_from_pickles(results_directory, results_folder_stub, file_suffix, n_batches=20, scenarios = s1)
	s2_dict = get_results_from_pickles(results_directory, results_folder_stub, file_suffix, n_batches=20, scenarios = s2)

	B = len(s1_dict.keys())

	y_position = np.arange(0, B, 1)

	s1_bridges_sorted, s1_sobol_sorted = sort_by_performance(s1_dict.keys(), [s1_dict[b] for b in s1_dict.keys()])
	s2_sobol_sorted = [s2_dict[str(b)] for b in s1_bridges_sorted]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.barh(y_position, s1_sobol_sorted, label='$S = 20$', color = 'cyan', alpha = 0.5, align = 'center')
	ax.barh(y_position, s2_sobol_sorted, label='$S = 39$', color = 'magenta', alpha = 0.5, align = 'center')
	ax.set_xlabel('Total-order Sobol index')
	ax.set_ylabel('Bridge ID')
	ax.tick_params(axis='y', which='major', labelsize=10)
	ax.set_yticks(y_position)
	ax.set_yticklabels(s1_bridges_sorted)
	ax.set_xscale('log')
	plt.legend(loc='best')
	plt.savefig('figs/sft2_sobol_comparison_log')

def plot_joint_damage_tracker(dx_sf, B = 20, S = 20, D = 10, N = 200, x = True):

	fig_folder = 'figs/'
	county = 'sft2'

	n_samples = N

	joint_damage_tracker_sf_testbed = np.zeros((B, B))

	for i in range(0,n_samples): # for each sample
		sample = dx_sf[:,:,i]
		for j in range(0,S*D):
		  for b1 in range(0,B):
				for b2 in range(0,B):
					if sample[j,b1] + sample[j,b2] > 1:
						joint_damage_tracker_sf_testbed[b1,b2] += 1

	assert check_symmetric(joint_damage_tracker_sf_testbed), 'Error -- the joint damage tracker is not symmetric, though it should be.'

	# so that things plot ok, set diagonal values to 0 (otherwise they are too large and mess up the colormap)
	joint_damage_tracker_plot = copy.deepcopy(joint_damage_tracker_sf_testbed)

	for b in range(0, B):  # get the diagonal values first
		temp = dx_sf[:, b, :]  # get the portion of the damage_tracker that gives us info for bridge b
		diagonal = np.sum(np.sum(temp))
		joint_damage_tracker_sf_testbed[b, b] = diagonal

	joint_damage_tracker_plot = copy.deepcopy(joint_damage_tracker_sf_testbed)

	sf_testbed = get_bridge_ids()
	ticklabels = [int(sf_testbed[i]) for i in range(0,B)]

	for b in range(0,B):
		joint_damage_tracker_plot[b,b] = 0

	y_position = np.arange(0,B,1)

	x_total = S * D * N  # array of size (SxD, B, N)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.xticks(rotation=90)
	mappable = ax.imshow(joint_damage_tracker_plot/x_total, cmap='Reds', interpolation='nearest') # visualize using a heatmap
	ax.set_xticks(y_position)
	ax.set_yticks(y_position)
	ax.set_xticklabels(ticklabels)
	ax.set_yticklabels(ticklabels)
	ax.tick_params(axis='y', which='major', labelsize=6)
	ax.tick_params(axis='x', which='major', labelsize=6)
	cbar = fig.colorbar(mappable) # probability that bridge i and bridge j are both damaged
	plt.savefig(fig_folder + county + '_joint_damage_x_s' + str(S) + '.png',bbox_inches='tight')

def get_joint_damage_unretrofitted(S = 20, D = 10, N = 1):

	dam_maps_per_scenario = D
	scenarios = S
	n_samples = N

	bridge_ids = get_bridge_ids()

	B = len(bridge_ids)
	n_bridges = B

	bridge_dict = get_dict('sft2_dict')

	if S == 20:
		map_indices_input = 'sobol_input/sft2_s20_final_map_indices.pkl'  # S = 20
		map_weights_input = 'sobol_input/sft2_s20_final_map_weights.pkl'  # S = 20
	elif S == 39:
		map_indices_input = 'sobol_input/sft2_s40_final_map_indices.pkl'  # S = 39
		map_weights_input = 'sobol_input/sft2_s40_final_map_weights.pkl'  # S = 39

	with open(map_indices_input, 'rb') as f:
		map_indices = pickle.load(f)
	with open(map_weights_input, 'rb') as f:
		map_weights = pickle.load(f)

	map_indices = map_indices
	map_weights = map_weights

	print 'map indices = ', map_indices
	print 'map weights = ', map_weights

	with open('sobol_input/U_temp.pkl', 'rb') as f:
		U_temp = pickle.load(f)

	if scenarios < 1992:  # get subset of uniform random numbers that correspond to the scenarios of interest
		U = np.zeros((scenarios * dam_maps_per_scenario, n_bridges))
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

	print 'U shapes (set and subset): ', U_temp.shape, U.shape

	undamaged_stats = load_undamaged_stats()

	no_damage_travel_time = undamaged_stats[0]
	no_damage_vmt = undamaged_stats[1]
	no_damage_trips_made = undamaged_stats[2]

	demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
							 'input/superdistricts_centroids_dummies.csv')

	# # Keep track of which bridges get damaged when computing f_X.
	damage_tracker = np.zeros((scenarios * dam_maps_per_scenario, n_bridges, n_samples))  # array of size (SxD, B, N)
	bridge_indices = {bridge_ids[i]: i for i in
					  range(0, len(bridge_ids))}  # each bridge has an index in the damage_tracker array

	X = np.zeros((B,))
	for i in range(0,B):
		index = bridge_indices[bridge_ids[i]]
		X[index] = bridge_dict[bridge_ids[i]]['ext_lnSa']

	dx_sf = cbs.run_traffic_model_dam_only(bridge_ids, bridge_dict, map_indices, map_weights, X, U, demand, damage_tracker, bridge_indices,
						  no_damage_travel_time, no_damage_vmt, no_damage_trips_made, num_gm_maps = 10, num_damage_maps = 3)

	fig_folder = 'figs/'
	county = 'sft2'

	joint_damage_tracker_sf_testbed = np.zeros((B, B))

	for i in range(0, n_samples):  # for each sample
		sample = dx_sf[:, :, i]
		for j in range(0, S * D):
			for b1 in range(0, B):
				for b2 in range(0, B):
					if sample[j, b1] + sample[j, b2] > 1:
						joint_damage_tracker_sf_testbed[b1, b2] += 1

	assert check_symmetric(
		joint_damage_tracker_sf_testbed), 'Error -- the joint damage tracker is not symmetric, though it should be.'

	# so that things plot ok, set diagonal values to 0 (otherwise they are too large and mess up the colormap)
	joint_damage_tracker_plot = copy.deepcopy(joint_damage_tracker_sf_testbed)

	for b in range(0, B):  # get the diagonal values first
		temp = dx_sf[:, b, :]  # get the portion of the damage_tracker that gives us info for bridge b
		diagonal = np.sum(np.sum(temp))
		joint_damage_tracker_sf_testbed[b, b] = diagonal

	joint_damage_tracker_plot = copy.deepcopy(joint_damage_tracker_sf_testbed)

	sf_testbed = get_bridge_ids()
	ticklabels = [int(sf_testbed[i]) for i in range(0, B)]

	for b in range(0, B):
		joint_damage_tracker_plot[b, b] = 0

	y_position = np.arange(0, B, 1)

	x_total = S * D * N  # array of size (SxD, B, N)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.xticks(rotation=90)
	mappable = ax.imshow(joint_damage_tracker_plot / x_total, cmap='Reds',
						 interpolation='nearest')  # visualize using a heatmap
	ax.set_xticks(y_position)
	ax.set_yticks(y_position)
	ax.set_xticklabels(ticklabels)
	ax.set_yticklabels(ticklabels)
	ax.tick_params(axis='y', which='major', labelsize=6)
	ax.tick_params(axis='x', which='major', labelsize=6)
	cbar = fig.colorbar(mappable)  # probability that bridge i and bridge j are both damaged
	plt.savefig(fig_folder + county + '_joint_damage_unret_s' + str(S) + '.png', bbox_inches='tight')

def plot_sobol_index_histogram():

	with open('sf_sobol_dict_tot_cost.pkl','rb') as f:
		result_dict = pickle.load(f)

	fig_folder = 'figs/'
	S = [result_dict[bridge] for bridge in result_dict.keys()]
	logS = [np.log(result_dict[bridge]) for bridge in result_dict.keys()]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(S, bins=10)
	plt.show()

def plot_cost_histogram(f_X_costs): # for one sample of the fragility function parameter vector, f

	print f_X_costs[10,:].shape

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(f_X_costs[10,:], bins=10)
	plt.show()

def get_retrofit_results(output_folder, n_retrofits, n_scenarios, dam_maps_per_scenario, filename='_sf_full',print_results=True): #TODO -- original get_retrofit_results() method -- does not correct expectation computation


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

	tt0, vmt0, trips0 = load_individual_undamaged_stats()

	print 'f_X_times.shape = ', f_X_times.shape

	# print the expected network performance
	if print_results:
		print 'for R = ', n_retrofits, ' expected travel times = ', f_X_avg_time, f_X_avg_time-tt0, alpha*(f_X_avg_time-tt0)/3600
		print 'for R = ', n_retrofits, ' expected trips made = ', f_X_avg_trip, trips0-f_X_avg_trip, beta*(trips0-f_X_avg_trip)
		print 'for R = ', n_retrofits, ' expected indirect costs = ', f_X_exp_indirect_cost*24*125 #24*125*(alpha*(f_X_avg_time-tt0)/3600 + beta*(trips0-f_X_avg_trip))
		# print 24*125*(alpha*(f_X_avg_time-tt0)/3600 + beta*(trips0-f_X_avg_trip)) # should be the same as f_X_exp_indirect_cost
		print 'for R = ', n_retrofits, ' expected direct costs = ', f_X_exp_direct_cost
		# print f_X_exp_direct_cost + (f_X_exp_cost-f_X_exp_direct_cost)
		print 'for R = ', n_retrofits, ' expected total cost = ', f_X_exp_cost #, f_X_exp_indirect_cost*24*125+f_X_exp_direct_cost

	# print 'for R = ', n_retrofits, ' expected conn cost = ', 78*8*max(0,trips0-f_X_avg_trip)
	# print 'for R = ', n_retrofits, ' expected delay cost = ',48*max(0, (f_X_avg_time-tt0)/3600)

	#plot_joint_damage_tracker_retrofits(damage_tracker[:,:,0], B = 20, S = n_scenarios, D = dam_maps_per_scenario, N = 1, name = 'r' + str(n_retrofits) + '_old')
	#plot_joint_damage_tracker_retrofits(damage_tracker[:,:,5], B = 20, S = n_scenarios, D = dam_maps_per_scenario, N = 1, name = 'r' + str(n_retrofits) + '_sobol')

	#
	# if n_retrofits > 0 and n_retrofits < 20:
	# 	print 'oldest trips', f_X_avg_trip[0], f_X_avg_trip[5] - f_X_avg_trip[0]
	# 	print 'weakest trips ', f_X_avg_trip[1], f_X_avg_trip[5] - f_X_avg_trip[1]
	# 	print 'busiest trips ', f_X_avg_trip[2], f_X_avg_trip[5] - f_X_avg_trip[2]
	# 	print 'composite trips ', f_X_avg_trip[3], f_X_avg_trip[5] - f_X_avg_trip[3]
	# 	print 'OAT trips', f_X_avg_trip[4], f_X_avg_trip[5] - f_X_avg_trip[4]
	# 	print 'sobol trips', f_X_avg_trip[5]
	# 	print 'difference = ', f_X_avg_trip[5] - f_X_avg_trip[0]

	# return f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, f_X_exp_indirect_cost

	return f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, f_X_exp_indirect_cost*24*125, f_X_exp_direct_cost, f_X_exp_cost

# def get_retrofit_results(output_folder, n_retrofits, n_scenarios, dam_maps_per_scenario, filename='_sf_full',print_results=True): #TODO -- correct expectation computation
#
# 	scenarios = n_scenarios
# 	batch_size = 1
#
# 	# store the results
# 	fX_times_output = output_folder + 'fX_times' + filename  # travel times for f_X
# 	fX_trips_output = output_folder + 'fX_trips' + filename  # trips made for f_X
# 	fX_vmts_output = output_folder + 'fX_vmts' + filename  # VMTs for f_X
# 	fX_avg_times_output = output_folder + 'fX_avg_time' + filename  # average TT
# 	fX_avg_trips_output = output_folder + 'fX_avg_trips' + filename  # average trips made
# 	fX_avg_vmts_output = output_folder + 'fX_avg_vmts' + filename  # average VMT
# 	fX_delay_costs_output = output_folder + 'fX_delay_costs' + filename
# 	fX_conn_costs_output = output_folder + 'fX_conn_costs' + filename
# 	fX_indirect_costs_output = output_folder + 'fX_indirect_costs' + filename
# 	fX_direct_costs_output = output_folder + 'fX_direct_costs' + filename
# 	fX_exp_indirect_cost_output = output_folder + 'fX_exp_indirect_costs' + filename
# 	fX_exp_direct_cost_output = output_folder + 'fX_exp_direct_costs' + filename
# 	fX_expected_cost_output = output_folder + 'fX_exp_costs' + filename
# 	#
# 	# damage_x_output = output_folder + 'damage_x' + filename
# 	#
# 	# # save data for f_X
# 	# with open(damage_x_output, 'rb') as f:
# 	# 	damage_tracker = pickle.load(f)
#
# 	with open(fX_times_output, 'rb') as f:  # save raw performance data
# 		f_X_times = pickle.load(f)
# 	with open(fX_trips_output, 'rb') as f:
# 		f_X_trips = pickle.load(f)
# 	with open(fX_vmts_output, 'rb') as f:
# 		f_X_vmts = pickle.load(f)
#
# 	with open(fX_avg_times_output, 'rb') as f:  # save average (expected) performance data
# 		f_X_avg_time = pickle.load(f)
# 	with open(fX_avg_trips_output, 'rb') as f:
# 		f_X_avg_trip = pickle.load(f)
# 	with open(fX_avg_vmts_output, 'rb') as f:
# 		f_X_avg_vmt = pickle.load(f)
#
# 	with open(fX_delay_costs_output, 'rb') as f:
# 		f_X_delay_costs = pickle.load(f)
# 	with open(fX_conn_costs_output, 'rb') as f:
# 		f_X_conn_costs = pickle.load(f)
# 	with open(fX_direct_costs_output, 'rb') as f:
# 		f_X_direct_costs = pickle.load(f)
# 	with open(fX_indirect_costs_output, 'rb') as f:
# 		f_X_indirect_costs = pickle.load(f)
#
# 	with open(fX_exp_direct_cost_output, 'rb') as f:
# 		f_X_exp_direct_cost = pickle.load(f)
# 	with open(fX_exp_indirect_cost_output, 'rb') as f:
# 		f_X_exp_indirect_cost = pickle.load(f)
# 	with open(fX_expected_cost_output, 'rb') as f:
# 		f_X_exp_cost = pickle.load(f)
#
# 	print 'f_X_times.shape', f_X_times.shape
#
# 	# Get the weighted average of all metrics of interest using the updated calculation and raw results.
# 	tt0, vmt0, trips0 = load_individual_undamaged_stats()
#
# 	if scenarios == 30:
# 		map_indices_input = 'sobol_input/sf_fullr_training_map_indices.pkl'  # S = 30 for training sf_fullr
# 		map_weights_input = 'sobol_input/sf_fullr_training_map_weights.pkl'  # S = 30 for training sf_fullr
# 	elif scenarios == 45:
# 		map_indices_input = 'sobol_input/sf_fullr_testing_map_indices.pkl'  # S = 30 for training sf_fullr
# 		map_weights_input = 'sobol_input/sf_fullr_testing_map_weights.pkl'  # S = 30 for training sf_fullr
# 	else:
# 		print 'Need 30 or 45 scenarios.'
#
# 	with open(map_indices_input, 'rb') as f:
# 		map_indices = pickle.load(f)
#
# 	with open(map_weights_input, 'rb') as f:
# 		map_weights = pickle.load(f)
#
# 	if len(map_indices) != scenarios:
# 		map_indices = map_indices[0]
# 		map_weights = map_weights[0]
#
# 	## GB: this gets hazard-consistent maps that we created from Miller's subsetting procedure
# 	sa_matrix_full = util.read_2dlist('input/sample_ground_motion_intensity_maps_road_only_filtered.txt',
# 									  delimiter='\t')
# 	sa_matrix = [sa_matrix_full[i] for i in
# 				 map_indices]  # GB: get the ground_motions for just the scenarios we are interested in
#
# 	lnsas = []
# 	magnitudes = []
# 	for row in sa_matrix:
# 		lnsas.append([log(float(sa)) for sa in row[4:]])
# 		magnitudes.append(float(row[2]))
#
# 	temp_fX_avg_times = np.zeros((batch_size,))
# 	temp_fX_avg_vmts = np.zeros((batch_size,))
# 	temp_fX_avg_trips = np.zeros((batch_size,))
# 	temp_fX_exp_indirect_cost = np.zeros((batch_size,))
# 	temp_fX_exp_direct_cost = np.zeros((batch_size,))
# 	temp_fX_expected_cost = np.zeros((batch_size,))
#
# 	for k in range(0, batch_size):
# 		# print '*** batch = ', i, ' sample = ', k
# 		average_travel_time, average_vmt, average_trips_made, average_direct_cost, average_delay_cost, average_connectivity_cost, \
# 		average_indirect_cost = compute_weighted_average_performance(lnsas, map_weights, num_damage_maps=10,
# 																	 travel_times=f_X_times[k, :],
# 																	 vmts=f_X_vmts[k, :],
# 																	 trips_made=f_X_trips[k, :],
# 																	 no_damage_travel_time=tt0,
# 																	 no_damage_vmt=vmt0,
# 																	 no_damage_trips_made=trips0,
# 																	 direct_costs=f_X_direct_costs[k, :])
#
# 		temp_fX_avg_times[k] = average_travel_time
# 		temp_fX_avg_vmts[k] = average_vmt
# 		temp_fX_avg_trips[k] = average_trips_made
# 		temp_fX_exp_direct_cost[k] = average_direct_cost
# 		temp_fX_exp_indirect_cost[k] = average_indirect_cost  # hourly
# 		temp_fX_expected_cost[k] = 24 * 125 * average_indirect_cost + average_direct_cost
#
# 	assert np.any(temp_fX_exp_indirect_cost == 0) == False, 'Error in correcting fX_exp_indirect_cost.'
# 	assert np.any(temp_fX_expected_cost == 0) == False, 'Error in correcting fX_expected_cost.'
#
#
# 	# # print the expected network performance
# 	# if print_results:
# 	# 	print 'for R = ', n_retrofits, ' expected travel times = ', f_X_avg_time, f_X_avg_time-tt0, alpha*(f_X_avg_time-tt0)/3600
# 	# 	print 'for R = ', n_retrofits, ' expected trips made = ', f_X_avg_trip, trips0-f_X_avg_trip, beta*(trips0-f_X_avg_trip)
# 	# 	print 'for R = ', n_retrofits, ' expected indirect costs = ', f_X_exp_indirect_cost*24*125 #24*125*(alpha*(f_X_avg_time-tt0)/3600 + beta*(trips0-f_X_avg_trip))
# 	# 	# print 24*125*(alpha*(f_X_avg_time-tt0)/3600 + beta*(trips0-f_X_avg_trip)) # should be the same as f_X_exp_indirect_cost
# 	# 	print 'for R = ', n_retrofits, ' expected direct costs = ', f_X_exp_direct_cost
# 	# 	# print f_X_exp_direct_cost + (f_X_exp_cost-f_X_exp_direct_cost)
# 	# 	print 'for R = ', n_retrofits, ' expected total cost = ', f_X_exp_cost #, f_X_exp_indirect_cost*24*125+f_X_exp_direct_cost
#
#
# 	# return f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, f_X_exp_indirect_cost*24*125, f_X_exp_direct_cost, f_X_exp_cost
# 	return temp_fX_avg_times, temp_fX_avg_vmts, temp_fX_avg_trips, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, temp_fX_exp_indirect_cost*24*125, temp_fX_exp_direct_cost, temp_fX_expected_cost
#

def print_retrofit_results(n_retrofits, n_scenarios):

	if n_scenarios == 48:
		output_folder = 'sobol_output/retrofits_total/s48/r' + str(n_retrofits) + '/'
	else:
		output_folder = 'sobol_output/retrofits_total/s30/r' + str(n_retrofits) + '/'

	dam_maps_per_scenario = 10

	f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, \
	f_X_exp_indirect_cost = get_retrofit_results(output_folder, n_retrofits, n_scenarios, dam_maps_per_scenario)

	return f_X_avg_time, f_X_avg_vmt, f_X_avg_trip, f_X_delay_costs, f_X_conn_costs, f_X_indirect_costs, f_X_exp_indirect_cost

def map_bridges():

	county = 'sf_fullr'
	county_dict, _ = get_sf_fullr_dict()

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

	i = 0
	for b in county_dict.keys():  # for every bridge (referenced by original ID)
		new_id = str(county_dict[b]['new_id'])  # get its new ID
		bridges.append(b)
		lat.append(bridge_dict_nbi[new_id]['lat'])
		long.append(bridge_dict_nbi[new_id]['long'])
		f.append(county_dict[b]['ext_lnSa'])
		age.append(bridge_dict_nbi[new_id]['age'])
		traffic.append(bridge_dict_nbi[new_id]['traffic'])

		i += 1

	# format the hover text for the map -- automatically includes lat/long; also includes bridge ID (original), fragility fxn parameter
	newline = '<br>'
	text = []

	for i in range(0, len(bridges)):
		text.append('original ID: ' + bridges[i] + newline +
					'f = ' + str(f[i]) + newline +
					'age = ' + str(age[i]) + newline +
					'daily traffic = ' + str(traffic[i]))

	map_center_lat = 37.7799
	map_center_long = -122.2822

	data = [go.Scattermapbox(lat=lat, lon=long, mode='markers', # for size
							 marker=go.Marker(size=10, color='red',
											  showscale=False, symbol='circle'),
							 text=text, name=' bridges'), ]


	layout = go.Layout(autosize=True, hovermode='closest', showlegend=False,
					   mapbox=go.layout.Mapbox(accesstoken=mapbox_access_token, bearing=0,
											   center=go.layout.mapbox.Center(
												   lat=map_center_lat,
												   lon=map_center_long),
											   zoom=11), )  # center at Napa County, zoom in
	fig = go.Figure(data=data, layout=layout)

	map_name = 'sf_fullr_map'
	plotly.offline.plot(fig, filename=map_name + '.html')

def plot_first_and_total_order(cost='total', plot_limit = 71, fig_folder = 'figs/'):

	bridge_dict, bridges = get_sf_fullr_dict()
	B = len(bridges)

	# get total-order result
	results_directory = 'sobol_output/run_sf_fullr/'
	results_folder_stub = 'run_sf_'
	n_batches = 37 # originally batches with 5 samples each; lost 3 because of time-out
	n_scenarios = 30 # S= 30 for training, S = 45 for testing
	print '****** sf_fullr results ******'
	sobol_index_dict_to = get_results_from_pickles(results_directory, results_folder_stub,
																			n_batches, max_batch = 40,
																			scenarios=n_scenarios, cost='total', retrofit=False, batch_size = 5)

	# get first-order results
	results_directory = 'sobol_output/run_sf_fullr_fo/'
	results_folder_stub = 'run_sf_'
	n_batches = 37  # originally batches with 5 samples each; lost 3 because of time-out
	n_scenarios = 30  # S= 30 for training, S = 45 for testing
	print '****** sf_fullr results ******'
	sobol_index_dict_fo = get_results_from_pickles(results_directory, results_folder_stub,
												n_batches, max_batch=40,
												scenarios=n_scenarios, cost='total', retrofit=False, batch_size=5,
												first_order=True)

	total_order_sobol = [sobol_index_dict_to[b] for b in bridges]
	# SORT BRIDGES BY TOTAL COST AND PLOT
	bridges_sorted, total_order_sorted = sort_by_performance(bridges, total_order_sobol)
	first_order_sorted = [sobol_index_dict_to[str(b)] for b in bridges_sorted]
	diff_sorted = [total_order_sorted[b] - first_order_sorted[b] for b in range(0,B)]

	# Stacked bar chart for total one-day costs for Napa County split up into components -- costs due to delays and costs due to lost trips (connectivity loss).
	y_position = np.arange(0, B, 1)
	fig = plt.figure()  # This is a plot of the simple OAT analysis results (i.e. not using scenarios, just using baseline performance computation).
	ax = fig.add_subplot(111)
	ax.set_xlabel('Cost, 2020 USD')
	ax.set_ylabel('ID of Damaged Bridge')
	ax.barh(y_position[0:plot_limit], first_order_sorted[B - plot_limit:B], align='center', alpha=1,
			label='first-order Sobol index', color='#f3c7c4')
	ax.barh(y_position[0:plot_limit], diff_sorted[B - plot_limit:B], align='center', alpha=1,
			label='total-order Sobol index',
			left=np.asarray(first_order_sorted[B - plot_limit:B]), color='#872435')
	ax.tick_params(axis='y', which='major', labelsize=6)
	ax.set_yticks(y_position[0:plot_limit])
	ax.set_yticklabels(bridges_sorted[B - plot_limit:B])
	ax.set_xscale('log')
	ax.legend(loc='best', prop={'size': 10})
	ax.set_title('First- and total-order Sobol indices')
	plt.savefig(fig_folder + 'sf_fullr_Sobol_index_components', bbox_inches='tight')

def plot_prioritized_bridge_characteristics(n_priorities = 10):


	bridge_dict, bridge_ids = get_sf_fullr_dict()

	fig = plt.figure()
	ax = fig.add_subplot(131)
	ax.hist([bridge_dict[b]['repair'] for b in bridge_ids])
	ax.set_xlabel('Repair cost')
	ax1 = fig.add_subplot(132)
	ax1.hist([bridge_dict[b]['age'] for b in bridge_ids])
	ax1.set_xlabel('Age')
	ax2 = fig.add_subplot(133)
	ax2.hist([bridge_dict[b]['traffic'] for b in bridge_ids])
	ax2.set_xlabel('Traffic')
	plt.savefig('sf_fullr_prioritized_characteristics.png', bbox_inches='tight')
	# plt.show()

def histogram_of_sobol_indices():
	with open('sf_fullr_sobol_dict_total_no_ret.pkl', 'rb') as f:
		sobol_index_dict = pickle.load(f)

	sobol_index_dict = sobol_index_dict[0]

	vals = [sobol_index_dict[b] for b in sobol_index_dict.keys()]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel('Normalized estimated total-order Sobol index, $\\hat{\\overline{S}}_b^2$, based on $\\mathbb{E}[C]$')
	ax.set_ylabel('Count')
	ax.hist(vals,color='y')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	plt.legend(loc='best', frameon=False)
	plt.savefig('sf_fullr_histogram_sobol_indices_total_cost_no_retrofit.png',bbox_inches='tight')



# histogram_of_sobol_indices()


# # Check sf_fullr training set (S = 30) results.
# results_directory = 'sobol_output/run_sf_fullr/'
# results_folder_stub = 'run_sf_'
# n_batches = 37 # originally batches with 5 samples each; lost 3 because of time-out
# n_scenarios = 30 # S= 30 for training, S = 45 for testing
# print '****** sf_fullr results ******'
# sobol_index_dict = get_results_from_pickles(results_directory, results_folder_stub,
# 																		n_batches, max_batch = 40,
# 																		scenarios=n_scenarios, cost='total', retrofit=False, batch_size = 5)
# with open('sf_fullr_sobol_dict_total_no_ret.pkl','wb') as f:
# 	pickle.dump(sobol_index_dict, f)


# # Get OAT results for bridges in sf_fullr.
# bridge_dict, bridge_ids = get_sf_fullr_dict()
#
# with open('/Users/gitanjali/Desktop/TransportNetworks/quick_traffic_model_GROUP/sobol_files_sf_full/sobol_output/oat_dict.pkl', 'rb') as f:
# 	oat_dict = pickle.load(f)
#
# for b in bridge_ids: # prints bridge original ID, total cost from OAT< indirect cost from OAT, direct cost (repair cost) from OAT
# 	print b, oat_dict[b]['exp_ind_cost'] + oat_dict[b]['exp_direct_cost'], oat_dict[b]['exp_ind_cost'], oat_dict[b]['exp_direct_cost']

# # Check sf_fullr FIRST-ORDER training set (S = 30) results.
# results_directory = 'sobol_output/run_sf_fullr_fo_all/'
# results_folder_stub = 'run_sf_'
# n_batches = 73 # originally 40 batches with 5 samples each; lost 3 because of time-out; then added another 20 batches, but lost 4, so added 16; 37 + 16 = 53; then added another 20 batches, so 73
# n_scenarios = 30 # S= 30 for training, S = 45 for testing
# print '****** sf_fullr results ******'
# sobol_index_dict = get_results_from_pickles(results_directory, results_folder_stub,
# 																		n_batches, max_batch = 80,
# 																		scenarios=n_scenarios, cost='total', retrofit=True, p=False, first_order=True, batch_size = 5)
#
# #

# bridge_dict, bridge_ids = get_sf_fullr_dict()
#
# for b in bridge_ids:
# 	print b, bridge_dict[b]['ext_lnSa'], bridge_dict[b]['ext_lnSa']*bridge_dict[b]['omega']

# # CHECK how many scenarios the training and testing sets have in common. (Answer is 19).
# map_indices_input = 'sobol_input/sf_fullr_testing_map_indices.pkl'  # S = 48
# map_indices_training_input = 'sobol_input/sf_fullr_training_map_indices.pkl'
#
# with open(map_indices_training_input, 'rb') as f:
# 	training = pickle.load(f)
#
# with open(map_indices_input, 'rb') as f:
# 	testing = pickle.load(f)
#
# training_set = set(training)
# testing_set = set(testing)
# print testing_set.intersection(training_set)


# # Check sf_fullr training set (S = 30) EXPANDED results.
# results_directory = 'sobol_output/run_sf_fullr_total_all/'
# results_folder_stub = 'run_sf_'
# n_batches = 54 # originally 37; then added 20 but lost 3 due to time-out
# n_scenarios = 30 # S= 30 for training, S = 45 for testing
# print '****** sf_fullr results ******'
# sobol_index_dict = get_results_from_pickles(results_directory, results_folder_stub,
# 																		n_batches, max_batch = 60,
# 																		scenarios=n_scenarios, cost='total', retrofit=False, batch_size = 5)
# with open('sf_fullr_sobol_dict_total_no_ret_54batches.pkl','wb') as f:
# 	pickle.dump(sobol_index_dict, f)

# # MAPPING Sobol' indices.


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # results_directory = 'sobol_output/run_sf_fullr_total_all/'
# # results_folder_stub = 'run_sf_'
# # n_batches = 54 # originally 37; then added 20 but lost 3 due to time-out
# # n_scenarios = 30 # S= 30 for training, S = 45 for testing
# # print '****** sf_fullr results ******'
# # sobol_index_dict, _, _, _, _ = get_results_from_pickles(results_directory, results_folder_stub,
# # 																		n_batches, max_batch = 60,
# # 																		scenarios=n_scenarios, cost='total', retrofit=False, batch_size = 5)
# #
# # with open('sobol_index_dict_54_batches.pkl', 'wb') as f:
# # 	pickle.dump(sobol_index_dict, f)
#
# with open('sobol_index_dict_54_batches.pkl', 'rb') as f:
# 	sobol_index_dict = pickle.load(f)
#
# S = [sobol_index_dict[bridge] for bridge in sobol_index_dict.keys()]
# for k, v in sobol_index_dict.items():
# 	print k, v
#
# # # scaled_dict = scale_sobol_indices(sobol_index_dict)
# # # scaled_S = [scaled_dict[bridge] for bridge in sobol_index_dict.keys()]
# # # logS = [np.log(sobol_index_dict[bridge]) for bridge in sobol_index_dict.keys()]
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.hist(S, bins = 8, color='r')
# # # ax.hist(scaled_S)
# # ax.set_xlabel('Normalized total-order Sobol index, $\\hat{\\overline{S}}_b^2$')
# # ax.set_ylabel('Count')
# # plt.savefig('sf_fullr_total_order_index_hist.png',bbox_inches='tight')
# #
# # # plot_bridge_sobol_indices('sf_fullr_total_order_index_map', sobol_index_dict)
#
# # sobol_index_dict = sobol_index_dict[0]
#
# vals = [sobol_index_dict[b] for b in sobol_index_dict.keys()]
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlabel('Normalized estimated total-order Sobol index, $\\hat{\\overline{S}}_b^2$, based on $\\mathbb{E}[C]$')
# ax.set_ylabel('Count')
# ax.hist(vals,color='r')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# plt.legend(loc='best', frameon=False)
# plt.savefig('figs_paper_final/sf_fullr_exp_cost_sobol_hist.png',bbox_inches='tight')



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# results_directory = 'sobol_output/run_sf_fullr_total_all/'
# results_folder_stub = 'run_sf_'
# n_batches = 37 # originally 37; then added 20 but lost 3 due to time-out
# n_scenarios = 30 # S= 30 for training, S = 45 for testing
# print '****** sf_fullr results ******'
# sobol_index_dict, _, _, _, _ = get_results_from_pickles(results_directory, results_folder_stub,
# 																		n_batches, max_batch = 40,
# 																		scenarios=n_scenarios, cost='total', retrofit=False, batch_size = 5)
# with open('sf_fullr_total_all_sobol_dict.pkl','wb') as f:
# 	pickle.dump(sobol_index_dict,f)

# # Compare how Sobol' indices vary with bridges' ages, fragilities, and daily average traffic volume.
# bridge_dict, bridge_ids = get_sf_fullr_dict()
#
# print bridge_dict[bridge_ids[0]].keys()
#
# age = [bridge_dict[bridge_id]['age'] for bridge_id in bridge_ids]
# traffic = [bridge_dict[bridge_id]['traffic']/1000 for bridge_id in bridge_ids]
# fragility = [bridge_dict[bridge_id]['ext_lnSa'] for bridge_id in bridge_ids]
# retrofit_fragility = [bridge_dict[bridge_id]['ext_lnSa']*bridge_dict[bridge_id]['omega'] for bridge_id in bridge_ids]
# sobol = [sobol_index_dict[bridge_id] for bridge_id in bridge_ids]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(221) # age
# ax1.scatter(age, sobol)
# ax1.set_xlabel('Age [years]')
# ax1.set_ylabel('Sobol index')
# ax2 = fig.add_subplot(222) # fragility
# ax2.scatter(fragility, sobol)
# ax2.set_xlabel('Fragility')
# ax3 = fig.add_subplot(223) # traffic
# ax3.scatter(traffic, sobol)
# ax3.set_xlabel('Traffic [thousands of cars]')
# ax3.set_ylabel('Sobol index')
# ax4 = fig.add_subplot(224) # fragility
# ax4.scatter(retrofit_fragility, sobol)
# ax4.set_xlabel('Retrofitted fragility')
# plt.savefig('characteristics_vs_sobol_scatter.png',bbox_inches = 'tight')
# plt.show()



def main():

	# #
	# # RECOMPUTE TOTAL-ORDER SOBOL INDICES based on revised average computation. #todo
	import os
	print os.getcwd()
	results_directory = 'sobol_output/run_sf_fullr_total_all/'
	results_folder_stub = 'run_sf_'
	n_batches =  92 # originally 37; then added 20 but lost 3 due to time-out; MAX OF 92
	max_batches = 140
	n_scenarios = 30 # S= 30 for training, S = 45 for testing
	print '****** sf_fullr results ******'
	# get_results_from_pickles_ORIGINAL(results_directory, results_folder_stub,n_batches, max_batch = 80,scenarios=n_scenarios, cost='total', retrofit=False, batch_size = 5)
	# sobol_index_dict, _, _, _, _ = get_raw_results_from_pickles(results_directory, results_folder_stub, n_batches, max_batch=max_batches, scenarios=n_scenarios, cost='total', retrofit=False, batch_size=5) # total cost, not including retrofit cost
	sobol_index_dict, _, _, _, _ = get_results_from_pickles(results_directory, results_folder_stub, n_batches, max_batch=max_batches, scenarios=n_scenarios, cost='total', retrofit=False, batch_size=5) # total cost, not including retrofit cost
	# plot_bridge_sobol_indices('sf_fullr_final', sobol_index_dict)
