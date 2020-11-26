import pickle, copy, csv
import numpy as np
import compute_bridge_sobol_sf_full as cbs
from itertools import permutations
import mahmodel_road_only as mahmodel
import bd_test as bd
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from compute_bridge_sobol import compute_sample_variance, compute_CI, compute_sample_mean
from collections import OrderedDict

def import_retrofit_list(csv_filepath):

	strategy = []

	with open(csv_filepath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		count = 0
		for row in reader:
			# print row[0:]
			s = [x for x in row[0:] if x]  # filter out empty cells in the csv
			#print s
			strategy.append(s[0])
			count += 1
		print 'count = ', count

	return strategy

def make_incremental_retrofit_lists(retrofit_list):

	ret_lists = []
	for i in range(1,len(retrofit_list)+1):
		ret_lists.append(retrofit_list[0:i])

	return ret_lists

def import_retrofit_tests(dict_name, csv_filepath): # these are tests for network effects, not of the efficacy of different strategies

	strategy = []

	with open(csv_filepath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		count = 0
		for row in reader:
			# print row[0:]
			s = [x for x in row[0:] if x] # filter out empty cells in the csv
			print s
			strategy.append(s)
			count += 1
		print 'count = ', count

	keys = [str(i) for i in range(0,count)]
	strategy_ordered = OrderedDict(zip(keys,strategy))

	with open(dict_name + '.pkl', 'wb') as f:
		pickle.dump(strategy_ordered, f)

	# with open('retrofit_tests_sf_fullr.pkl', 'wb') as f:
	# 	pickle.dump(strategy_ordered, f)

	return strategy_ordered

def import_retrofits(csv_filepath):

	strategy = {}

	with open(csv_filepath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		# next(reader)  # skip header
		count = 0
		for row in reader:
			strategy[row[0]] = row[1:-1]
			#strategy[row[0]].append(row[2:-1])
			count += 1
		print 'count = ', count

	strategy_ordered = OrderedDict(strategy)

	# with open('sobol_input/retrofit_strategies_sf_fullr.pkl','wb') as f:
	# 	pickle.dump(strategy_ordered,f)

	return strategy_ordered

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

def make_retrofit_samples_sf(n_retrofits, retrofit_lists, n_bridges, partial_dict, bridge_ids,retrofit_sample_filepath, omega='unique', custom_filename=None):
	# Make samples of fragility function parameters for different sets of retrofits based on age, weakness, traffic volume,
	# and total-order Sobol index.

	n_retrofit_strategies = len(retrofit_lists)

	# STEP 5c. Generate the lists of fragility parameters for each retrofitting strategy test. Each set of fragility
	# parameters will later be passed as the input 'x' to mahmodel_road_only_napa_2.py.

	retrofit_x = np.zeros((n_retrofit_strategies, n_bridges)) # fragility parameters for bridges retrofitted per each retrofit strategy

	r_counts = np.zeros((n_retrofit_strategies,))
	if n_retrofits > 0:
		i = 0 # counter for bridge index
		for b in bridge_ids:  # for the bridges in the case study
			ff_og = partial_dict[str(b)]['ext_lnSa']  # original fragility function parameter

			if omega == 'unique':
				ff_ret = ff_og*partial_dict[str(b)]['omega']
			elif omega == 'uniform':
				ff_ret = ff_og * 2  # retrofitted fragility function parameter

			for r in range(0, n_retrofit_strategies): # for each retrofit strategy
				#print b, retrofit_lists[r][0:n_retrofits]
				if b in retrofit_lists[r][0:n_retrofits]:
					print b in retrofit_lists[r][0:n_retrofits]
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
		# retrofit_sample_file = 'retrofit_samples_r_' + str(n_retrofits) + '.pkl'
		retrofit_sample_file = 'retrofit_' + str(n_retrofits) + '.pkl'

	retrofit_sample_filepath = retrofit_sample_filepath +retrofit_sample_file

	with open(retrofit_sample_filepath, 'wb') as f:
		pickle.dump(retrofit_samples, f)

	print 'stored at: ', retrofit_sample_filepath

	print 'Done generating samples for retrofit strategies with R = ', n_retrofits, '.'

def make_retrofit_test_samples(max_n_retrofits, retrofit_dict, n_bridges, partial_dict, bridge_ids,retrofit_sample_filepath, omega='unique', custom_filename=None):
	# Make samples of fragility function parameters for different sets of retrofits based on age, weakness, traffic volume,
	# and total-order Sobol index.

	n_retrofit_strategies = len(retrofit_dict.keys())

	# STEP 5c. Generate the lists of fragility parameters for each retrofitting strategy test. Each set of fragility
	# parameters will later be passed as the input 'x' to mahmodel_road_only_napa_2.py.

	retrofit_x = np.zeros((n_retrofit_strategies, n_bridges)) # fragility parameters for bridges retrofitted per each retrofit strategy

	r_counts = np.zeros((n_retrofit_strategies,))
	if max_n_retrofits > 0:
		i = 0 # counter for bridge index
		for b in bridge_ids:  # for the bridges in the case study
			ff_og = partial_dict[str(b)]['ext_lnSa']  # original fragility function parameter

			if omega == 'unique':
				ff_ret = ff_og*partial_dict[str(b)]['omega']
			elif omega == 'uniform':
				ff_ret = ff_og * 2  # retrofitted fragility function parameter

			for r in range(0, n_retrofit_strategies): # for each retrofit strategy
				#print b, retrofit_lists[r][0:n_retrofits]
				if b in retrofit_dict[str(r)]:
					print b in retrofit_dict[str(r)]
					retrofit_x[r, i] = ff_ret
					r_counts[r] += 1
				else:
					retrofit_x[r, i] = ff_og
			i += 1
	elif max_n_retrofits == 0:
		retrofit_x = np.zeros((1,n_bridges))
		for i in range(0,n_bridges):
			retrofit_x[0,i] = partial_dict[str(bridge_ids[i])]['ext_lnSa']
	else:
		'Number of retrofits is not valid.'

	print r_counts

	assert np.sum(r_counts) == np.sum([len(retrofit_dict[str(i)]) for i in range(0,len(retrofit_dict.keys()))]), 'The number of retrofits specified by the user was not carried out.'

	# Store the fragility function parameter samples in a single array and save as a pickle.
	retrofit_samples = np.asarray(retrofit_x)

	print 'retrofit samples'
	print retrofit_samples

	if custom_filename is not None:
		retrofit_sample_file = custom_filename
	else:
		retrofit_sample_file = 'retrofit_test_samples.pkl'

	retrofit_sample_filepath = retrofit_sample_filepath +retrofit_sample_file

	with open(retrofit_sample_filepath, 'wb') as f:
		pickle.dump(retrofit_samples, f)

	print 'stored at: ', retrofit_sample_filepath

	print 'Done generating samples for retrofit tests.'

def make_retrofit_test_samples_series(max_n_retrofits, retrofit_lists, n_bridges, partial_dict, bridge_ids,retrofit_sample_filepath, omega='unique', custom_filename=None):
	# Make samples of fragility function parameters for different sets of retrofits based on age, weakness, traffic volume,
	# and total-order Sobol index.

	n_retrofit_strategies = len(retrofit_lists)

	# STEP 5c. Generate the lists of fragility parameters for each retrofitting strategy test. Each set of fragility
	# parameters will later be passed as the input 'x' to mahmodel_road_only_napa_2.py.

	retrofit_x = np.zeros((n_retrofit_strategies, n_bridges)) # fragility parameters for bridges retrofitted per each retrofit strategy

	r_counts = np.zeros((n_retrofit_strategies,))
	if max_n_retrofits > 0:
		i = 0 # counter for bridge index
		for b in bridge_ids:  # for the bridges in the case study
			ff_og = partial_dict[str(b)]['ext_lnSa']  # original fragility function parameter

			if omega == 'unique':
				ff_ret = ff_og*partial_dict[str(b)]['omega']
			elif omega == 'uniform':
				ff_ret = ff_og * 2  # retrofitted fragility function parameter

			for r in range(0, n_retrofit_strategies): # for each retrofit strategy
				#print b, retrofit_lists[r][0:n_retrofits]
				if b in retrofit_lists[r]:
					# print b in retrofit_lists[r]
					retrofit_x[r, i] = ff_ret
					r_counts[r] += 1
				else:
					retrofit_x[r, i] = ff_og
			i += 1
	elif max_n_retrofits == 0:
		retrofit_x = np.zeros((1,n_bridges))
		for i in range(0,n_bridges):
			retrofit_x[0,i] = partial_dict[str(bridge_ids[i])]['ext_lnSa']
	else:
		'Number of retrofits is not valid.'

	print r_counts

	assert np.sum(r_counts) == np.sum([len(r) for r in retrofit_lists]), 'The number of retrofits specified by the user was not carried out.'

	# Store the fragility function parameter samples in a single array and save as a pickle.
	retrofit_samples = np.asarray(retrofit_x)

	print 'retrofit samples'
	print retrofit_samples

	if custom_filename is not None:
		retrofit_sample_file = custom_filename
	else:
		retrofit_sample_file = 'retrofit_test_samples.pkl'

	retrofit_sample_filepath = retrofit_sample_filepath +retrofit_sample_file

	with open(retrofit_sample_filepath, 'wb') as f:
		pickle.dump(retrofit_samples, f)

	print 'stored at: ', retrofit_sample_filepath

	print 'Done generating samples for retrofit tests.'


# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # B = len(bridge_ids)
# # ret_strats = import_retrofits('sf_fullr_2020 - full_strategies.csv')
# # strategies = ['oldest', 'busiest', 'weakest', 'composite', 'OAT total', 'OAT indirect', 'OAT direct', 'Sobol exp. total', 'Sobol exp. indirect', 'Sobol exp. direct']
# # # for r in strategies: # manual check that import was done correctly
# # # 	print r, ret_strats[r][0:3]
# #
# # # Create list of lists -- bridges to retrofit for each strategy in variable strategies (i.e. in that order)
# # retrofit_strategies = []
# # for r in strategies:
# # 	retrofit_strategies.append(ret_strats[r])
# #
# # # for r in range(0,len(strategies)): # another manual check
# # # 	print retrofit_strategies[r][0:3]
# #
# # # Generate retrofit samples for sf_fullr --
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # for i in range(0,len(bridge_ids)):
# # 	make_retrofit_samples_sf(n_retrofits=i, retrofit_lists=retrofit_strategies,n_bridges=B,partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique')
# #
# # r_all = [bridge_ids]
# # make_retrofit_samples_sf(n_retrofits=B, retrofit_lists=r_all, n_bridges=B, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique')
#
# # master_dict = get_master_dict()
# # new_to_old = {}
# # for b in master_dict.keys():
# # 	new = master_dict[b]['new_id']
# # 	new_to_old[new] = b
# #
# # for b in new_to_old.keys():
# # 	print b, new_to_old[b]
#
# # # RETROFIT TESTS: Trying to detect and quantify network effects by retrofitting different combinations of bridges -- from Google Doc sf_fullr_2020.
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - retrofit_tests_export (1).csv'
# # tests = import_retrofit_tests(retrofit_test_filepath)
# # print tests.keys()
# # # # Generate retrofit samples for sf_fullr
# # # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # # make_retrofit_test_samples(max_n_retrofits=3, retrofit_dict=tests, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_tests.pkl')
#
#
# # # RETROFIT TESTS: Trying to detect and quantify network effects by retrofitting different combinations of bridges -- from Google Doc sf_fullr_2020.
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - ret_test_sobol.csv'
# # dict_name = 'ret_test_sf_fullr_sobol'
# # tests = import_retrofit_tests(dict_name,retrofit_test_filepath)
# # # for k, v in tests.items():
# # # 	print k, v
# # # with open(dict_name+'.pkl','rb') as f:
# # # 	strategies = pickle.load(f)
# # # for k, v in strategies.items():
# # # 	print k,v
# # # Generate retrofit samples for sf_fullr
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # make_retrofit_test_samples(max_n_retrofits=3, retrofit_dict=tests, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_tests_sobol.pkl')
#
# # # RETROFIT TESTS: retrofit each bridge individually.
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - individual_rets.csv'
# # dict_name = 'ind_rets_sf_fullr'
# # tests = import_retrofit_tests(dict_name,retrofit_test_filepath)
# # # for k, v in tests.items():
# # # 	print k, v
# # # with open(dict_name+'.pkl','rb') as f:
# # # 	strategies = pickle.load(f)
# # # for k, v in strategies.items():
# # # 	print k,v
# # # Generate retrofit samples for sf_fullr
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # make_retrofit_test_samples(max_n_retrofits=1, retrofit_dict=tests, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_ind.pkl')
#
# # # RETROFIT TESTS: retrofit each bridge according to the OAT strategy
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - oat_rets.csv'
# # dict_name = 'ind_rets_sf_fullr'
# # tests = import_retrofit_list(retrofit_test_filepath)
# # ret_lists = make_incremental_retrofit_lists(tests)
# # # Generate retrofit samples for sf_fullr
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # make_retrofit_test_samples_series(max_n_retrofits=len(bridge_ids), retrofit_lists=ret_lists, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_oat.pkl')
#
# # # RETROFIT TESTS: retrofit each bridge according to the age strategy
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - age_rets.csv'
# # dict_name = 'ind_rets_sf_fullr'
# # tests = import_retrofit_list(retrofit_test_filepath)
# # ret_lists = make_incremental_retrofit_lists(tests)
# # # Generate retrofit samples for sf_fullr
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # make_retrofit_test_samples_series(max_n_retrofits=len(bridge_ids), retrofit_lists=ret_lists, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_age.pkl')
#
# # # RETROFIT TESTS: retrofit each bridge according to the OAT strategy
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - oat_rets.csv'
# # dict_name = 'ind_rets_sf_fullr'
# # tests = import_retrofit_list(retrofit_test_filepath)
# # ret_lists = make_incremental_retrofit_lists(tests)
# # ret_lists = ret_lists[0:10]
# # # Generate retrofit samples for sf_fullr
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # make_retrofit_test_samples_series(max_n_retrofits=10, retrofit_lists=ret_lists, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_oat_short.pkl')
#
# # # RETROFIT TESTS: retrofit each bridge according to the traffic-based strategy
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - traffic_rets.csv'
# # # dict_name = 'ind_rets_sf_fullr'
# # tests = import_retrofit_list(retrofit_test_filepath)
# # ret_lists = make_incremental_retrofit_lists(tests)
# # # Generate retrofit samples for sf_fullr
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # make_retrofit_test_samples_series(max_n_retrofits=10, retrofit_lists=ret_lists, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_traffic.pkl')
# #
# # # RETROFIT TESTS: retrofit each bridge according to the composite-based strategy.
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - composite_rets.csv'
# # # dict_name = 'ind_rets_sf_fullr'
# # tests = import_retrofit_list(retrofit_test_filepath)
# # ret_lists = make_incremental_retrofit_lists(tests)
# # # Generate retrofit samples for sf_fullr
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # make_retrofit_test_samples_series(max_n_retrofits=10, retrofit_lists=ret_lists, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_composite.pkl')
# #
# # # RETROFIT TESTS: retrofit each bridge according to the fragility-based strategy.
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - fragility_rets.csv'
# # # dict_name = 'ind_rets_sf_fullr'
# # tests = import_retrofit_list(retrofit_test_filepath)
# # ret_lists = make_incremental_retrofit_lists(tests)
# # # Generate retrofit samples for sf_fullr
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # make_retrofit_test_samples_series(max_n_retrofits=10, retrofit_lists=ret_lists, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_fragility.pkl')
#
# # # RETROFIT TESTS: retrofit each bridge according to the Sobol' index-based strategy with respect to the 98.5th percentile of the total cost.
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020_risk_aversion - retrofit strategy 985.csv'
# # # dict_name = 'ind_rets_sf_fullr'
# # tests = import_retrofit_list(retrofit_test_filepath)
# # ret_lists = make_incremental_retrofit_lists(tests)
# # # Generate retrofit samples for sf_fullr
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # make_retrofit_test_samples_series(max_n_retrofits=10, retrofit_lists=ret_lists, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_sobol_985.pkl')
#
#
# # # RETROFIT TESTS: retrofit each bridge according to the Sobol-index-based strategy -- batched into 10 groups of 7 to make computation faster
# # retrofit_test_filepath = 'sobol_input/sf_fullr_2020 - sobol_rets.csv'
# # dict_name = 'ind_rets_sf_fullr'
# # tests = import_retrofit_list(retrofit_test_filepath)
# # ret_lists = make_incremental_retrofit_lists(tests)
# # # ret_lists = ret_lists[0:10]
# # # Generate retrofit samples for sf_fullr
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# # batch_size = 7
# # n_batches = 10
# # for i in range(0,n_batches):
# # 	max_n_retrofits = (i+1)*batch_size
# # 	make_retrofit_test_samples_series(max_n_retrofits=max_n_retrofits, retrofit_lists=ret_lists[i*batch_size:(i+1)*batch_size], n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_sobol_batch_'+ str(i)+'.pkl')
# #
#
#
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # # make_retrofit_samples_sf(n_retrofits=5, retrofit_lists=[['1027','1066','947','976','977']],n_bridges=71,partial_dict=bridge_dict,bridge_ids=bridge_ids,retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_sobol_test_5.pkl')
# # # make_retrofit_samples_sf(n_retrofits=6, retrofit_lists=[['1027','1066','947','976','977','971']],n_bridges=71,partial_dict=bridge_dict,bridge_ids=bridge_ids,retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_sobol_test_6.pkl')
# # make_retrofit_samples_sf(n_retrofits=3, retrofit_lists=[['976','977','971']],n_bridges=71,partial_dict=bridge_dict,bridge_ids=bridge_ids,retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_sobol_test_3.pkl')
# # make_retrofit_samples_sf(n_retrofits=2, retrofit_lists=[['971','977']],n_bridges=71,partial_dict=bridge_dict,bridge_ids=bridge_ids,retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_sobol_test_2a.pkl')
# # make_retrofit_samples_sf(n_retrofits=2, retrofit_lists=[['976','971']],n_bridges=71,partial_dict=bridge_dict,bridge_ids=bridge_ids,retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_sobol_test_2b.pkl')
#
# # # MAKE retrofit samples based on max cost Sobol' indices (total-order, not including retrofit cost).
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # make_retrofit_samples_sf(n_retrofits=3, retrofit_lists=[['951','1029','993']],n_bridges=71,partial_dict=bridge_dict,bridge_ids=bridge_ids,retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_max_cost_3.pkl')
# # make_retrofit_samples_sf(n_retrofits=2, retrofit_lists=[['951','1029',]],n_bridges=71,partial_dict=bridge_dict,bridge_ids=bridge_ids,retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_max_cost_2.pkl')
# # make_retrofit_samples_sf(n_retrofits=1, retrofit_lists=[['951']],n_bridges=71,partial_dict=bridge_dict,bridge_ids=bridge_ids,retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_max_cost_1.pkl')
# #
# # # MAKE retrofit samples based on 95th percentile total-order Sobol' indices (total cost, not including retrofit cost).
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_risk_aversion - retrofit strategy 95th.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_95p_'+str(len(r))+'.pkl')
#
# # # MAKE retrofit samples based on 99th percentile total-order Sobol' indices (total cost, not including retrofit cost).
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_risk_aversion - retrofit strategy 99th.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # # for r in pruned_ret_lists:
# # # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_99p_'+str(len(r))+'.pkl')
#
#
# # # MAKE retrofit samples based on 99th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_risk_aversion - retrofit strategy 99th 100 samples.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_99p100_'+str(len(r))+'.pkl')
#
# # # MAKE retrofit samples based on 99th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_risk_aversion - retrofit strategy 995th percentile.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_995p_'+str(len(r))+'.pkl')
#
# # # MAKE retrofit samples based on 98.5th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_risk_aversion - retrofit strategy 985.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_985p_'+str(len(r))+'.pkl')
#
# # # MAKE retrofit samples based on 98.5th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_risk_aversion - retrofit strategy 998.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_998p_'+str(len(r))+'.pkl')
#
# # # MAKE retrofit samples based on 98.5th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_risk_aversion - retrofit strategy 9995.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_9995p_'+str(len(r))+'.pkl')
#
# # # MAKE retrofit samples based on 99.7th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_risk_aversion - retrofit strategy 997.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_997p_'+str(len(r))+'.pkl')
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # MAKE retrofit samples based on 99.0th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list('sobol_input/risk_aversion_ret_strats_370_samples/sf_fullr_2020_risk_aversion_370_samples - ret_strat_99p.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# # 							 bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_ra_370/',
# # 							 omega='unique',custom_filename='retrofit_99p_'+str(len(r))+'.pkl')
#
#
# # # MAKE retrofit samples based on 99.9th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list('sobol_input/risk_aversion_ret_strats_370_samples/sf_fullr_2020_risk_aversion_370_samples - ret_strat_999p.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# # 							 bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_ra_370/',
# # 							 omega='unique',custom_filename='retrofit_999p_'+str(len(r))+'.pkl')
# #
# # # MAKE retrofit samples based on 99.95th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list('sobol_input/risk_aversion_ret_strats_370_samples/sf_fullr_2020_risk_aversion_370_samples - ret_strat_9995p.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# # 							 bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_ra_370/',
# # 							 omega='unique',custom_filename='retrofit_9995p_'+str(len(r))+'.pkl')
#
#
# # # MAKE retrofit samples based on 99.5th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list('sobol_input/risk_aversion_ret_strats_370_samples/sf_fullr_2020_risk_aversion_370_samples - ret_strat_995p.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# # 							 bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_ra_370/',
# # 							 omega='unique',custom_filename='retrofit_995p_'+str(len(r))+'.pkl')
# #
# # # MAKE retrofit samples based on 98.5th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list(
# # 	'sobol_input/risk_aversion_ret_strats_370_samples/sf_fullr_2020_risk_aversion_370_samples - ret_strat_985p.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# # 							 bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_ra_370/',
# # 							 omega='unique', custom_filename='retrofit_985p_' + str(len(r)) + '.pkl')
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # MAKE retrofit samples based on expected total cost Sobol' indices with low p.
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020 lowp and highp - retrofit lowp.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_lowp_'+str(len(r))+'.pkl')
#
# # # MAKE retrofit samples based on expected total cost Sobol' indices with high p.
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020 lowp and highp - retrofit highp.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/',omega='unique',custom_filename='retrofit_highp_'+str(len(r))+'.pkl')
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #
# # # MAKE retrofit samples based on expected total cost Sobol' indices with low p (CORRECTED).
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_diff_p corrected - ret_strat_lowp.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_p/',omega='unique',custom_filename='retrofit_lowp_'+str(len(r))+'.pkl')
# #
# # # MAKE retrofit samples based on expected total cost Sobol' indices with high p (CORRECTED).
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_diff_p corrected - ret_strat_regular.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_p/',omega='unique',custom_filename='retrofit_regular_'+str(len(r))+'.pkl')
# #
# #
# #
# # # MAKE retrofit samples based on expected total cost Sobol' indices with high p (CORRECTED).
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_diff_p corrected - ret_strat_highp.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_p/',omega='unique',custom_filename='retrofit_highp_'+str(len(r))+'.pkl')
# #
#
# # INVESTIGATING WEIRD RESULTS OF RETROFIT STRATEGIES BETWEEN R = 20 AND R = 40 for expected total cost.
#
# # # MAKE retrofit samples based on expected total cost Sobol' indices with low p (CORRECTED).
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_diff_p corrected - ret_strat_lowp.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i <= 20 or i >= 40:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_p/',omega='unique',custom_filename='retrofit_lowp_'+str(len(r))+'.pkl')
#
# # MAKE retrofit samples based on expected total cost Sobol' indices with regular p (CORRECTED).
# retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_diff_p corrected - ret_strat_regular.csv')
# all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# pruned_ret_lists = []
# i = 1
# for r in all_ret_lists:
# 	if i <= 20 or i >= 40:
# 		pass
# 	else:
# 		pruned_ret_lists.append(r)
# 	i += 1
#
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# for r in pruned_ret_lists:
# 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_p/',omega='unique',custom_filename='retrofit_regular_'+str(len(r))+'.pkl')
#
# #
# # # MAKE retrofit samples based on expected total cost Sobol' indices with high p (CORRECTED).
# # retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020_diff_p corrected - ret_strat_highp.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i <= 20 or i >= 40:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_p/',omega='unique',custom_filename='retrofit_highp_'+str(len(r))+'.pkl')
# #

#
# # # MAKE retrofit samples based on expected total cost + retrofit cost Sobol' indices with regular p.
# retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020 retrofit cost inclusion - retrofit_ret_cost.csv')
# all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# pruned_ret_lists = []
# i = 1
# for r in all_ret_lists:
# 	if i > 10 and i % 10 != 0 and i != 65:
# 		pass
# 	else:
# 		pruned_ret_lists.append(r)
# 	i += 1
#
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# for r in pruned_ret_lists:
# 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/retrofit_ret_cost/',omega='unique',custom_filename='retrofit_'+str(len(r))+'.pkl')

# # # MAKE retrofit samples based on (decrease in expected total cost)/retrofit cost Sobol' indices with regular p.
# retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020 retrofit cost inclusion - retrofit_ret_ratio.csv')
# all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# pruned_ret_lists = []
# i = 1
# for r in all_ret_lists:
# 	if i > 10 and i % 10 != 0 and i != 65:
# 		pass
# 	else:
# 		pruned_ret_lists.append(r)
# 	i += 1
#
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# for r in pruned_ret_lists:
# 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/retrofit_ret_ratio/',omega='unique',custom_filename='retrofit_'+str(len(r))+'.pkl')

# # # MAKE retrofit samples based on (decrease in expected total cost)/retrofit cost Sobol' indices with regular p.
# retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020 retrofit cost inclusion - retrofit_ret_ratio_reverse.csv')
# all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# pruned_ret_lists = []
# i = 1
# for r in all_ret_lists:
# 	if i > 10 and i % 10 != 0 and i != 65:
# 		pass
# 	else:
# 		pruned_ret_lists.append(r)
# 	i += 1
#
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# for r in pruned_ret_lists:
# 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples/retrofit_ret_ratio_reverse/',omega='unique',custom_filename='retrofit_'+str(len(r))+'.pkl')

# # INVESTIGATING WEIRD RESULTS OF RETROFIT STRATEGIES BETWEEN R = 50 AND R = 70 for 985p.
# # # MAKE retrofit samples based on 98.5th percentile total-order Sobol' indices (total cost, not including retrofit cost), just the first N = 100 samples.
# # retrofit_list = import_retrofit_list(
# # 	'sobol_input/risk_aversion_ret_strats_370_samples/sf_fullr_2020_risk_aversion_370_samples - ret_strat_985p.csv')
# # all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# #
# # bridge_dict, bridge_ids = get_sf_fullr_dict()
# # for r in pruned_ret_lists:
# # 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# # 							 bridge_ids=bridge_ids, retrofit_sample_filepath='sobol_input/retrofit_samples_ra_370/',
# # 							 omega='unique', custom_filename='retrofit_985p_' + str(len(r)) + '.pkl')

# # Re-run all retrofit strategies of interest with revised averages. #TODO -- make retrofit samples based on strategies with revised averages
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# B = len(bridge_ids)
# ret_strats = import_retrofits('sobol_input/sf_fullr_2020 Sobol results revised averages - revised strategies.csv')
# strategies = ['oldest', 'busiest', 'weakest', 'composite', 'OAT total', 'Sobol exp. total']
# for r in strategies: # manual check that import was done correctly
# 	print r, ret_strats[r][0:3]
#
# # Create list of lists -- bridges to retrofit for each strategy in variable strategies (i.e. in that order)
# retrofit_strategies = []
# for r in strategies:
# 	retrofit_strategies.append(ret_strats[r])
#
# for r in range(0,len(strategies)): # another manual check
# 	print retrofit_strategies[r][0:3]
#
# # Generate retrofit samples for sf_fullr --
# retrofit_sample_filepath = 'sobol_input/retrofit_samples/retrofit_revised_avg/'
# for i in range(0,len(bridge_ids)):
# 	make_retrofit_samples_sf(n_retrofits=i, retrofit_lists=retrofit_strategies,n_bridges=B,partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique')
#
# r_all = [bridge_ids]
# make_retrofit_samples_sf(n_retrofits=B, retrofit_lists=r_all, n_bridges=B, partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique')

# # RETROFIT TESTS: retrofit each bridge according to the CORRECTED EXPECTATION Sobol-index-based strategy -- batched into 10 groups of 7 to make computation faster
# retrofit_test_filepath = 'sobol_input/sf_fullr_2020_Sobol_results_revised_averages_exp_cost.csv' # based on N = 370
# # dict_name = 'ind_rets_sf_fullr'
# tests = import_retrofit_list(retrofit_test_filepath)
# ret_lists = make_incremental_retrofit_lists(tests)
# for r in ret_lists:
# 	print r
# # Generate retrofit samples for sf_fullr
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# batch_size = 7
# n_batches = 10
# for i in range(0,n_batches):
# 	max_n_retrofits = (i+1)*batch_size
# 	print max_n_retrofits
# 	# make_retrofit_test_samples_series(max_n_retrofits=max_n_retrofits, retrofit_lists=ret_lists[i*batch_size:(i+1)*batch_size], n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_revised_avg_sobol_batch_'+ str(i)+'.pkl')

# # # RETROFIT TESTS -- retrofit each bridge according to the corrected expectation Sobol-index-based strategy, but using different numbers of samples to compute Sobol' indices.
# retrofit_list = import_retrofit_list('sobol_input/sf_fullr_2020 risk aversion results - 999p_n500.csv')
# all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# # pruned_ret_lists = []
# # i = 1
# # for r in all_ret_lists:
# # 	if i > 10 and i % 10 != 0 and i != 65:
# # 		pass
# # 	else:
# # 		pruned_ret_lists.append(r)
# # 	i += 1
# pruned_ret_lists = all_ret_lists
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# for r in pruned_ret_lists:
# 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# 							 bridge_ids=bridge_ids,
# 							 retrofit_sample_filepath='sobol_input/retrofit_samples/999p_n500/',
# 							 omega='unique',custom_filename='retrofit_'+str(len(r))+'.pkl')


# MAKE RETROFIT SAMPLES to test strategies that account for the cost of retrofits in the Sobol' index computation, with
# revised average computations.

# retrofit_list = import_retrofit_list(
# 	'sobol_input/sf_fullr_2020 budget results - ret_ratio ranking.csv')
# all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# pruned_ret_lists = []
# i = 1
# for r in all_ret_lists:
# 	if i > 10 and i % 10 != 0 and i != 65:
# 		pass
# 	else:
# 		pruned_ret_lists.append(r)
# 	i += 1
#
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# for r in pruned_ret_lists:
# 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# 							 bridge_ids=bridge_ids,
# 							 retrofit_sample_filepath='sobol_input/retrofit_samples/ret_ratio/',
# 							 omega='unique', custom_filename='retrofit_' + str(len(r)) + '.pkl')

# retrofit_list = import_retrofit_list(
# 	'sobol_input/sf_fullr_2020 budget results - ret_ratio_reverse ranking.csv')
# all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# pruned_ret_lists = []
# i = 1
# for r in all_ret_lists:
# 	if i > 10 and i % 10 != 0 and i != 65:
# 		pass
# 	else:
# 		pruned_ret_lists.append(r)
# 	i += 1
#
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# for r in pruned_ret_lists:
# 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# 							 bridge_ids=bridge_ids,
# 							 retrofit_sample_filepath='sobol_input/retrofit_samples/ret_ratio_reverse/',
# 							 omega='unique', custom_filename='retrofit_' + str(len(r)) + '.pkl')

# # MAKE RETROFIT SAMPLES for strategies based on samples constructed with different values of p -- lowp, regularp, and highp.
# # Each strategy is based on N = 190 samples.
# retrofit_list = import_retrofit_list(
# 	'sobol_input/sf_fullr_2020 sampling retrofit probabilities - lowp ranking.csv')
# all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# pruned_ret_lists = []
# i = 1
# for r in all_ret_lists:
# 	if i > 10 and i % 10 != 0 and i != 65:
# 		pass
# 	else:
# 		pruned_ret_lists.append(r)
# 	i += 1
#
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# for r in pruned_ret_lists:
# 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# 							 bridge_ids=bridge_ids,
# 							 retrofit_sample_filepath='sobol_input/retrofit_samples/retrofit_samples_lowp/',
# 							 omega='unique', custom_filename='retrofit_' + str(len(r)) + '.pkl')
#
# retrofit_list = import_retrofit_list(
# 	'sobol_input/sf_fullr_2020 sampling retrofit probabilities - highp ranking.csv')
# all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# pruned_ret_lists = []
# i = 1
# for r in all_ret_lists:
# 	if i > 10 and i % 10 != 0 and i != 65:
# 		pass
# 	else:
# 		pruned_ret_lists.append(r)
# 	i += 1
#
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# for r in pruned_ret_lists:
# 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# 							 bridge_ids=bridge_ids,
# 							 retrofit_sample_filepath='sobol_input/retrofit_samples/retrofit_samples_highp/',
# 							 omega='unique', custom_filename='retrofit_' + str(len(r)) + '.pkl')
#
# retrofit_list = import_retrofit_list(
# 	'sobol_input/sf_fullr_2020 sampling retrofit probabilities - regularp ranking.csv')
# all_ret_lists = make_incremental_retrofit_lists(retrofit_list)
# pruned_ret_lists = []
# i = 1
# for r in all_ret_lists:
# 	if i > 10 and i % 10 != 0 and i != 65:
# 		pass
# 	else:
# 		pruned_ret_lists.append(r)
# 	i += 1
#
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# for r in pruned_ret_lists:
# 	make_retrofit_samples_sf(n_retrofits=len(r), retrofit_lists=[r], n_bridges=71, partial_dict=bridge_dict,
# 							 bridge_ids=bridge_ids,
# 							 retrofit_sample_filepath='sobol_input/retrofit_samples/retrofit_samples_regularp/',
# 							 omega='unique', custom_filename='retrofit_' + str(len(r)) + '.pkl')



# # RETROFIT TESTS: retrofit each bridge according to the Sobol' index-based strategy with respect to the 99.9th percentile of the total cost, CORRECTED.
# retrofit_test_filepath = 'sobol_input/sf_fullr_2020 risk aversion results - 999p_n500.csv'
# # dict_name = 'ind_rets_sf_fullr'
# tests = import_retrofit_list(retrofit_test_filepath)
# ret_lists = make_incremental_retrofit_lists(tests)
# # Generate retrofit samples for sf_fullr
# bridge_dict, bridge_ids = get_sf_fullr_dict()
# retrofit_sample_filepath = 'sobol_input/retrofit_samples/'
# make_retrofit_test_samples_series(max_n_retrofits=len(bridge_ids), retrofit_lists=ret_lists, n_bridges=len(bridge_ids),partial_dict=bridge_dict, bridge_ids=bridge_ids, retrofit_sample_filepath=retrofit_sample_filepath, omega='unique', custom_filename='retrofit_sobol_999p_n500.pkl')
#
