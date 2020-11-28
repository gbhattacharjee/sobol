# Author: Gitanjali Bhattacharjee
# Purpose: Test different retrofit strategies on bridges by getting samples of fragility
# function parameters for each retrofit strategy and running them through Miller's modified traffic model.


import pickle
import numpy as np
from compute_bridge_sobol_sf_full import precompute_network_performance
import mahmodel_road_only as mahmodel
import bd_test as bd


def get_bridge_ids(ints=False):
    with open('input/sf_fullr_bridge_ids.pkl', 'rb') as f:
        bridge_ids = pickle.load(f)
    if ints:
        bridge_ids = [int(b) for b in bridge_ids]

    return bridge_ids

def get_sf_fullr_dict():

	with open('input/sf_fullr_dict.pkl', 'rb') as f:
		sf_dict = pickle.load(f)

	with open('input/sf_fullr_bridge_ids.pkl', 'rb') as f:
		bridge_ids = pickle.load(f)

	return sf_dict, bridge_ids


def main(output_folder, n_retrofits, n_scenarios, dam_maps_per_scenario, retrofit_sample_file, filename='_sf_fullr'):
    # This function takes as input:
    # - output_folder, the name of an output folder (string) -- should be formatted as 'folder/'
    # - n_retrofits, a number of retrofits to allocate (int) -- can be 1, 2, or 3 -- equivalent to R
    # - n_scenarios, a number of scenarios to consider (int) -- can be 19 or 50 -- equivalent to S
    # - dam_maps_per_scenario, the number of damage maps to generate for each scenario -- equivalent to D

    partial_dict, bridge_ids = get_sf_fullr_dict()

    # sf_testbed = [951, 1081, 935, 895, 947, 993, 976, 898, 925, 926, 966, 917, 1065, 953, 972, 990, 899, 919, 904,
    #               940]  # see bridge_metadata_NBI_sf_tetsbed/sf_testbed_new_3 -- otherwise referred to as sf_testbed_2
    # bridge_ids = [str(b) for b in sf_testbed]

    # bridge_ids = get_bridge_ids()

    with open(retrofit_sample_file, 'rb') as f:
        retrofit_samples = pickle.load(f)

    print 'retrofit samples are: '
    print retrofit_samples

    # Get the indices and weights of the maps in the subset of the UCERF2 catalog on which we're testing. These maps and
    # weights were generated using Miller's subset optimization code in MATLAB and stored in
    # the MATLAB folder example_map_selection/napa_scenarios_subset_indices_testing.csv. The pickle versions were stored in
    # the Python project folder in both ground_motions and sobol_input.

    if n_scenarios == 45:
        map_indices_input = 'sobol_input/sf_fullr_testing_map_indices.pkl'  # S = 48
        map_weights_input = 'sobol_input/sf_fullr_testing_map_weights.pkl'  # S = 48

    with open(map_indices_input, 'rb') as f:
        map_indices = pickle.load(f)
    with open(map_weights_input, 'rb') as f:
        map_weights = pickle.load(f)

    map_indices = map_indices
    map_weights = map_weights

    print 'map indices = ', map_indices
    print 'map weights = ', map_weights

    # Set testing parameters.
    n_bridges = retrofit_samples.shape[1] # number of bridges we are considering
    n_samples = retrofit_samples.shape[0] # number of sets of fragility function parameters we are testing
    scenarios = len(map_indices) # number of scenarios to consider -- should be 50 -- also equal to number of ground-motion maps we're considering, since we are only consider 1 per scenario
    assert len(map_indices) == n_scenarios, 'The number of maps does not match the user-requested number of scenarios.'
    assert len(map_weights) == n_scenarios, 'The number of map weights does not match the user-requested number of scenarios.'

    n_evals = scenarios * dam_maps_per_scenario * n_samples

    print '****************************************************************************************'
    print 'Your settings will require ', n_evals, ' function evaluations (traffic model runs).'
    print 'N = ', n_samples  # number of samples of fragility function parameters
    print 'S = ', scenarios  # number of scenarios to consider -- equivalent to ground-motion maps since we are only considering 1 GM map per scenario
    print 'D = ', dam_maps_per_scenario
    print 'B = ', n_bridges
    print 'R = ', n_retrofits
    print '****************************************************************************************'

    with open('sobol_input/U_good_sf_fullr.pkl', 'rb') as f:
        U_temp = pickle.load(f)

    if scenarios < 1992:  # get subset of uniform random numbers that correspond to the scenarios of interest
        U = np.zeros((scenarios * dam_maps_per_scenario, n_bridges))
        i = 0
        for s in map_indices:
            U[i * dam_maps_per_scenario:(i + 1) * dam_maps_per_scenario, :] = U_temp[s * dam_maps_per_scenario:(
                                                                                                                           s + 1) * dam_maps_per_scenario,
                                                                              :]
            i += 1

    assert U.shape[0] == scenarios * dam_maps_per_scenario, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
    assert U.shape[1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'

    print 'U shapes (set and subset): ', U_temp.shape, U.shape

    # Set up traffic model and run it.
    G = mahmodel.get_graph()
    assert G.is_multigraph() == False, 'You want a directed graph without multiple edges between nodes'

    demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
                             'input/superdistricts_centroids_dummies.csv')

    no_damage_travel_time, no_damage_vmt, no_damage_trips_made, G = precompute_network_performance(
        G, demand)

    print 'no_damage_travel_time = ', no_damage_travel_time
    print 'no_damage_trips_made = ', no_damage_trips_made

    # # Keep track of which bridges get damaged when computing f_X.
    damage_tracker = np.zeros((scenarios*dam_maps_per_scenario,n_bridges,n_samples)) # array of size (SxD, B, N)
    bridge_indices = {bridge_ids[i]:i for i in range(0,len(bridge_ids))} # each bridge has an index in the damage_tracker array

    f_X_times = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_trips = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_vmts = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_delay_costs = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_conn_costs = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_indirect_costs = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_direct_costs = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))

    f_X_avg_time = np.zeros((n_samples,))
    f_X_avg_trip = np.zeros((n_samples,))
    f_X_exp_indirect_cost = np.zeros((n_samples,))
    f_X_exp_direct_cost = np.zeros((n_samples,))
    f_X_exp_cost = np.zeros((n_samples,))  # total expected cost
    f_X_avg_vmt = np.zeros((n_samples,))

    for i in range(0,n_samples):
        print 'Starting traffic model for sample ', i, ' of ', n_samples

        tt, trips, vmts, average_tt, average_trips, average_vmt, temp_damage_tracker, delay_costs, \
        connectivity_costs, indirect_costs, repair_costs, expected_delay_cost, expected_conn_cost, \
        expected_indirect_cost, expected_repair_cost, expected_total_cost, retrofit_cost = mahmodel.main(i, map_indices, map_weights,
                                                                                          bridge_ids,
                                                                                          partial_dict, retrofit_samples[i,:],
                                                                                          U, demand, damage_tracker,
                                                                                          bridge_indices,
                                                                                          no_damage_travel_time,
                                                                                          no_damage_vmt,
                                                                                          no_damage_trips_made,
                                                                                          num_gm_maps=n_scenarios, num_damage_maps=dam_maps_per_scenario)

        f_X_times[i, :] = tt
        f_X_trips[i, :] = trips
        f_X_vmts[i, :] = vmts

        # save raw cost data
        f_X_delay_costs[i, :] = delay_costs
        f_X_conn_costs[i, :] = connectivity_costs
        f_X_indirect_costs[i, :] = indirect_costs
        f_X_direct_costs[i, :] = repair_costs

        # save expected data
        f_X_avg_time[i] = average_tt
        f_X_avg_trip[i] = average_trips
        f_X_avg_vmt[i] = average_vmt
        f_X_exp_indirect_cost[i] = expected_indirect_cost # should be multiplied by 24 hours per day x 125 days of restoration time to get $
        f_X_exp_direct_cost[i] = expected_repair_cost
        #f_X_exp_cost[i] = expected_total_cost
        f_X_exp_cost[i] = 24*125*expected_indirect_cost + expected_repair_cost # 24 hours per day (assuming demand is the same all day), 125 days (mean restoration time for extensively damaged bridge per Shinozuka et al 2003)

        damage_tracker = temp_damage_tracker

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
    with open(damage_x_output, 'wb') as f:
        pickle.dump(damage_tracker, f)

    with open(fX_times_output, 'wb') as f:  # save raw performance data
        pickle.dump(f_X_times, f)
    with open(fX_trips_output, 'wb') as f:
        pickle.dump(f_X_trips, f)
    with open(fX_vmts_output, 'wb') as f:
        pickle.dump(f_X_vmts, f)

    with open(fX_avg_times_output, 'wb') as f:  # save average (expected) performance data
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

    # print the expected network performance
    print 'for R = ', n_retrofits, ' expected travel times = ', f_X_avg_time

    print 'for R = ', n_retrofits, ' expected indirect costs = ', f_X_exp_indirect_cost

    print 'for R = ', n_retrofits, ' expected direct costs = ', f_X_exp_direct_cost

    print 'for R = ', n_retrofits, ' expected total costs = ', f_X_exp_cost

    print 'retrofit samples: ', retrofit_samples

def main_for_tests(output_folder, n_scenarios, dam_maps_per_scenario, retrofit_sample_file, filename='_sf_fullr'):
    # This function takes as input:
    # - output_folder, the name of an output folder (string) -- should be formatted as 'folder/'
    # - n_retrofits, a number of retrofits to allocate (int) -- can be 1, 2, or 3 -- equivalent to R
    # - n_scenarios, a number of scenarios to consider (int) -- can be 19 or 50 -- equivalent to S
    # - dam_maps_per_scenario, the number of damage maps to generate for each scenario -- equivalent to D

    partial_dict, bridge_ids = get_sf_fullr_dict()

    # sf_testbed = [951, 1081, 935, 895, 947, 993, 976, 898, 925, 926, 966, 917, 1065, 953, 972, 990, 899, 919, 904,
    #               940]  # see bridge_metadata_NBI_sf_tetsbed/sf_testbed_new_3 -- otherwise referred to as sf_testbed_2
    # bridge_ids = [str(b) for b in sf_testbed]

    # bridge_ids = get_bridge_ids()

    with open(retrofit_sample_file, 'rb') as f:
        retrofit_samples = pickle.load(f)

    print 'retrofit samples are: '
    print retrofit_samples

    # Get the indices and weights of the maps in the subset of the UCERF2 catalog on which we're testing. These maps and
    # weights were generated using Miller's subset optimization code in MATLAB and stored in
    # the MATLAB folder example_map_selection/napa_scenarios_subset_indices_testing.csv. The pickle versions were stored in
    # the Python project folder in both ground_motions and sobol_input.

    if n_scenarios == 45:
        map_indices_input = 'sobol_input/sf_fullr_testing_map_indices.pkl'  # S = 48
        map_weights_input = 'sobol_input/sf_fullr_testing_map_weights.pkl'  # S = 48

    with open(map_indices_input, 'rb') as f:
        map_indices = pickle.load(f)
    with open(map_weights_input, 'rb') as f:
        map_weights = pickle.load(f)

    map_indices = map_indices
    map_weights = map_weights

    print 'map indices = ', map_indices
    print 'map weights = ', map_weights

    # Set testing parameters.
    n_bridges = retrofit_samples.shape[1] # number of bridges we are considering
    n_samples = retrofit_samples.shape[0] # number of sets of fragility function parameters we are testing
    scenarios = len(map_indices) # number of scenarios to consider -- should be 50 -- also equal to number of ground-motion maps we're considering, since we are only consider 1 per scenario
    assert len(map_indices) == n_scenarios, 'The number of maps does not match the user-requested number of scenarios.'
    assert len(map_weights) == n_scenarios, 'The number of map weights does not match the user-requested number of scenarios.'

    n_evals = scenarios * dam_maps_per_scenario * n_samples

    print '****************************************************************************************'
    print 'Your settings will require ', n_evals, ' function evaluations (traffic model runs).'
    print 'N = ', n_samples  # number of samples of fragility function parameters
    print 'S = ', scenarios  # number of scenarios to consider -- equivalent to ground-motion maps since we are only considering 1 GM map per scenario
    print 'D = ', dam_maps_per_scenario
    print 'B = ', n_bridges
    print 'R = varies'
    print '****************************************************************************************'

    with open('sobol_input/U_good_sf_fullr.pkl', 'rb') as f:
        U_temp = pickle.load(f)

    if scenarios < 1992:  # get subset of uniform random numbers that correspond to the scenarios of interest
        U = np.zeros((scenarios * dam_maps_per_scenario, n_bridges))
        i = 0
        for s in map_indices:
            U[i * dam_maps_per_scenario:(i + 1) * dam_maps_per_scenario, :] = U_temp[s * dam_maps_per_scenario:(
                                                                                                                           s + 1) * dam_maps_per_scenario,
                                                                              :]
            i += 1

    assert U.shape[0] == scenarios * dam_maps_per_scenario, 'Error -- the number of rows in U does not equal the number of damage maps, S*D.'
    assert U.shape[1] == n_bridges, 'Error -- the number of rows in U does not equal the number of bridges of interest, B.'

    print 'U shapes (set and subset): ', U_temp.shape, U.shape

    # Set up traffic model and run it.
    G = mahmodel.get_graph()
    assert G.is_multigraph() == False, 'You want a directed graph without multiple edges between nodes'

    demand = bd.build_demand('input/BATS2000_34SuperD_TripTableData.csv',
                             'input/superdistricts_centroids_dummies.csv')

    no_damage_travel_time, no_damage_vmt, no_damage_trips_made, G = precompute_network_performance(
        G, demand)

    print 'no_damage_travel_time = ', no_damage_travel_time
    print 'no_damage_trips_made = ', no_damage_trips_made

    # # Keep track of which bridges get damaged when computing f_X.
    damage_tracker = np.zeros((scenarios*dam_maps_per_scenario,n_bridges,n_samples)) # array of size (SxD, B, N)
    bridge_indices = {bridge_ids[i]:i for i in range(0,len(bridge_ids))} # each bridge has an index in the damage_tracker array

    f_X_times = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_trips = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_vmts = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_delay_costs = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_conn_costs = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_indirect_costs = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))
    f_X_direct_costs = np.zeros((n_samples, n_scenarios * dam_maps_per_scenario))

    f_X_avg_time = np.zeros((n_samples,))
    f_X_avg_trip = np.zeros((n_samples,))
    f_X_exp_indirect_cost = np.zeros((n_samples,))
    f_X_exp_direct_cost = np.zeros((n_samples,))
    f_X_exp_cost = np.zeros((n_samples,))  # total expected cost
    f_X_avg_vmt = np.zeros((n_samples,))

    for i in range(0,n_samples):
        print 'Starting traffic model for sample ', i, ' of ', n_samples

        tt, trips, vmts, average_tt, average_trips, average_vmt, temp_damage_tracker, delay_costs, \
        connectivity_costs, indirect_costs, repair_costs, expected_delay_cost, expected_conn_cost, \
        expected_indirect_cost, expected_repair_cost, expected_total_cost, retrofit_cost = mahmodel.main(i, map_indices, map_weights,
                                                                                          bridge_ids,
                                                                                          partial_dict, retrofit_samples[i,:],
                                                                                          U, demand, damage_tracker,
                                                                                          bridge_indices,
                                                                                          no_damage_travel_time,
                                                                                          no_damage_vmt,
                                                                                          no_damage_trips_made,
                                                                                          num_gm_maps=n_scenarios, num_damage_maps=dam_maps_per_scenario)

        f_X_times[i, :] = tt
        f_X_trips[i, :] = trips
        f_X_vmts[i, :] = vmts

        # save raw cost data
        f_X_delay_costs[i, :] = delay_costs
        f_X_conn_costs[i, :] = connectivity_costs
        f_X_indirect_costs[i, :] = indirect_costs
        f_X_direct_costs[i, :] = repair_costs

        # save expected data
        f_X_avg_time[i] = average_tt
        f_X_avg_trip[i] = average_trips
        f_X_avg_vmt[i] = average_vmt
        f_X_exp_indirect_cost[i] = expected_indirect_cost # should be multiplied by 24 hours per day x 125 days of restoration time to get $
        f_X_exp_direct_cost[i] = expected_repair_cost
        #f_X_exp_cost[i] = expected_total_cost
        f_X_exp_cost[i] = 24*125*expected_indirect_cost + expected_repair_cost # 24 hours per day (assuming demand is the same all day), 125 days (mean restoration time for extensively damaged bridge per Shinozuka et al 2003)

        damage_tracker = temp_damage_tracker

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
    with open(damage_x_output, 'wb') as f:
        pickle.dump(damage_tracker, f)

    with open(fX_times_output, 'wb') as f:  # save raw performance data
        pickle.dump(f_X_times, f)
    with open(fX_trips_output, 'wb') as f:
        pickle.dump(f_X_trips, f)
    with open(fX_vmts_output, 'wb') as f:
        pickle.dump(f_X_vmts, f)

    with open(fX_avg_times_output, 'wb') as f:  # save average (expected) performance data
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

    # print the expected network performance
    print 'expected travel times = ', f_X_avg_time

    print 'expected indirect costs = ', f_X_exp_indirect_cost

    print 'expected direct costs = ', f_X_exp_direct_cost

    print 'expected total costs = ', f_X_exp_cost

    print 'retrofit samples: ', retrofit_samples

