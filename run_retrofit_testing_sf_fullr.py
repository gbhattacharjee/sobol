import retrofit_testing
import time

n_retrofits = 0
n_scenarios = 45 # S = 30 for training, S = 45 for testing
output_folder = 'sobol_output/retrofits/r' + str(n_retrofits) + '/'
retrofit_sample_file = 'sobol_input/retrofit_samples/retrofit_samples_r_' + str(n_retrofits)+ '.pkl'
dam_maps_per_scenario = 10

start = time.time()
print 'Starting retrofit testing for sf_fullr...'
retrofit_testing.main(output_folder, n_retrofits, n_scenarios, dam_maps_per_scenario, retrofit_sample_file)
print 'Duration was ', time.time() - start, ' seconds for SF retrofit testing with B = 20, S = ', n_scenarios,' D = ', dam_maps_per_scenario, ' and R = ', n_retrofits