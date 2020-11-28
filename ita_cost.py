#Author: Mahalia Miller
#Date: Jan. 21, 2013

#this code does iterative travel assignment. 
import sys, util, pdb, math, time
import networkx as nx
import bd_test

iteration_vals = [0.4, 0.3, 0.2, 0.1] #assign od vals in this amount per iteration. These are recommeded values from the Nature paper, http://www.nature.com/srep/2012/121220/srep01001/pdf/srep01001.pdf
#iteration_vals = [1.0, 0, 0, 0]
# class ITA:
#   def __init__(self, G, demand):
#     #G is a networkx graph, it can be damaged. Edges need to have a free flow travel time, a capacity, a variable called flow (which we'll change to keep track of flows assigned to each link), and a variable called t_a (which we'll change based on the flows)
#     self.G = G
#     self.demand = demand
#
#   def assign(self):
#     #does 4 iterations in which it assigns od, updates t_a. find paths to minimize travel time and we record route for each od pair
#     total_flow = 0
#
#     # TEMPORARY GB ADDITIONS
#     not_made = []
#     not_made_od = []
#     no_count_1 = 0
#     no_count_2 = 0
#     # origins = ['1000028', '1000029', '1000024', '1000025', '1000026', '1000027', '1000020', '1000021', '1000008', '1000009', '1000006', '1000007', '1000004', '1000005', '1000002', '1000003', '1000001', '1000022', '1000030', '1000034', '1000033', '1000032', '1000011', '1000010', '1000013', '1000012', '1000015', '1000014', '1000017', '1000016', '1000019', '1000018', '1000023', '1000031']
#     # destinations = {}
#     # for origin in origins:
#     #   destinations[origin] = self.demand[origin].keys()
#
#     null_count = 0
#     total_count = 0
#     for edge in self.G.edges():
#         if self.G[edge[0]][edge[1]]['capacity'] <= 0:
#             null_count += 1
#         total_count += 1
#     print 'in ITA -- null count for graph self.G is = ', null_count, ' of ', total_count
#
#     for i in range(4): #do 4 iterations
#       #for origin in origins:
#       for origin in self.demand.keys(): #origin is an actual A (one end of an actual edge in the graph)
#         #find the shortest paths from this origin to each destination
#         paths_dict = nx.single_source_dijkstra_path(self.G, origin, cutoff = None, weight = 't_a') #Compute shortest path between source and all other reachable nodes for a weighted graph. Returns dict keyed by by target with the value being a list of node ids of the shortest path
#         for destination in self.demand[origin].keys(): #actual A or B (one end of an actual edge in the graph)
#         # for destination in destinations[origin]:
#           od_flow = iteration_vals[i] * self.demand[origin][destination] #to get morning flows, take 5.3% of daily driver values. 11.5/(4.5*6+11.5*10+14*4+4.5*4) from Figure S10 of http://www.nature.com/srep/2012/121220/srep01001/extref/srep01001-s1.pdf
#
#           #get path
#           path_list = paths_dict[destination] #list of nodes
#
#           od_made = True # start off by assuming we can make the trip from origin to destination along the shortest path as described in path_list
#           #increment flow on the paths and update t_a
#           for index in range(0, len(path_list) - 1):
#             u = path_list[index]
#             v = path_list[index + 1]
#
#             if self.G.is_multigraph():
#               num_multi_edges =  len(self.G[u][v]) #if not multigraph, this just returns the number of edge attributes
#               if num_multi_edges >1: #multi-edge
#                 #identify multi edge with lowest t_a
#                 best = 0
#                 best_t_a = float('inf')
#                 for multi_edge in self.G[u][v].keys():
#                   new_t_a = self.G[u][v][multi_edge]['t_a'] #causes problems
#                   if (new_t_a < best_t_a) and (self.G[u][v][multi_edge]['capacity']>0):
#                     best = multi_edge
#                     best_t_a = new_t_a
#               else:
#                 best = 0
#               if (self.G[u][v][best]['capacity']>0): # if the edge capacity is greater than 0
#                 self.G[u][v][best]['flow'] += od_flow # assign the flow to the edge
#                 t = util.TravelTime(self.G[u][v][best]['t_0'], self.G[u][v][best]['capacity'])
#                 travel_time = t.get_new_travel_time(self.G[u][v]['flow']) # CHANGE PER JML
#                 self.G[u][v][best]['t_a'] = travel_time
#               else:
#                 no_count_1 += 1
#             else:
#               try:
#                 if (self.G[u][v]['capacity']>0): # if the edge capacity is greater than 0
#                   self.G[u][v]['flow'] += od_flow # assign the flow to the edge
#                   t = util.TravelTime(self.G[u][v]['t_0'], self.G[u][v]['capacity'])
#                   travel_time= t.get_new_travel_time(self.G[u][v]['flow']) # CHANGE PER JML
#                   self.G[u][v]['t_a'] = travel_time #in seconds
#                 else:
#                   od_made = False # if we couldn't make the trip between node u and node v, change od_made to False
#                   not_made.append((u,v))
#                   not_made_od.append((origin,destination))
#                   no_count_2 += 1
#               except KeyError as e:
#                 print('found key error: ', e)
#                 pdb.set_trace()
#
#           total_flow += od_made*od_flow # if we manage to make it all the way from origin to destination along the path in path_list, add the od_flow to the total_flow
#
#     print 'not made: ', len(list(set(not_made))), list(set(not_made))
#     print 'no counts = ', no_count_1, no_count_2
#     #print 'not made OD: ', len(list(set(not_made_od))), list(set(not_made_od))
#
#     return self.G, total_flow #, total_dem, lost_flow, od_pairs_lost
#
#
# def test():
#   #create graph info
#   G = nx.MultiDiGraph()
#   G.add_node(1)
#   G.add_node(2)
#   G.add_node(3)
#   G.add_edge(1,2,capacity_0=1000,capacity=1000,t_0=15,t_a=15,flow=0, distance=10)
#   G.add_edge(1,2,capacity_0=3000,capacity=3000,t_0=20,t_a=20,flow=0, distance=10)
#   G.add_edge(2,3, capacity_0=0, capacity=0, t_0 = 10, t_a = 10, flow = 0, distance = 10)
#   #get od info. This is in format of a dict keyed by od, like demand[sd1][sd2] = 200000.
#   demand = {}
#   demand[1] = {}
#   demand[1][2] = 8000 #divide by 0.053 since that is what we multiply by above
#   demand[2] = {}
#   demand[2][3] = 4000
#
#   #call ita
#   it = ITA(G,demand)
#   newG, total_flow, total_demand, lost_flow, od_pairs_lost = it.assign()
#   print(newG)
#   print('total flow:', total_flow)
#   print('total demand:', total_demand)
#   print('proportion of demand met:', (total_flow/total_demand))
#   for n,nbrsdict in newG.adjacency_iter():
#     for nbr,keydict in nbrsdict.items():
#       for key,eattr in keydict.items():
#         print (n, nbr, eattr['flow'])
#   print('should have flow of 3200 and 4800')

class ITA:
  def __init__(self, G, demand):
    #G is a networkx graph, it can be damaged. Edges need to have a free flow travel time, a capacity, a variable called flow (which we'll change to keep track of flows assigned to each link), and a variable called t_a (which we'll change based on the flows)
    self.G = G
    self.demand = demand

  def assign(self):
    #does 4 iterations in which it assigns od, updates t_a. find paths to minimize travel time and we record route for each od pair

    trips_made = 0 # tracker for number of trips made on the whole network
    lost_trips = 0 # tracker for number of trips that don't get made

    # GB BUG FIX #1: sort OD pairs to fix inconsistency across different runs of the traffic assignment
    origins = [int(i) for i in self.demand.keys()] # get SD node IDs as integers
    origins.sort() # sort them
    origins = [str(i) for i in origins] # make them strings again

    od_dict = bd_test.build_od(self.demand)  # GB BUG FIX #1, continued: sort OD pairs to fix inconsistency across different runs of the traffic assignment

    for i in range(len(iteration_vals)): #do 4 iterations
      for origin in origins:
        paths_dict = nx.single_source_dijkstra_path(self.G, origin, cutoff = None, weight = 't_a') #Compute shortest path between source and all other reachable nodes for a weighted graph. Returns dict keyed by by target with the value being a list of node ids of the shortest path

        for destination in od_dict[origin]:
          od_flow = 0.053*iteration_vals[i] * self.demand[origin][destination] #to get morning flows, take 5.3% of daily driver values. 11.5/(4.5*6+11.5*10+14*4+4.5*4) from Figure S10 of http://www.nature.com/srep/2012/121220/srep01001/extref/srep01001-s1.pdf

          try:
            # GB NOTE: get shortest path from origin to destination, but note that it may not exist -- that's why this is in a "try/except" framework
            path_list = paths_dict[destination] #list of nodes

            od_made = True # start off by assuming we can make the trip from origin to destination along the shortest path as described in path_list
            for index in range(0, len(path_list) - 1): # GB BUG FIX #2: before assigning flow to edges, check whether the trip between origin and destination can be completed -- if it can't, we assume people won't attempt to make it and get stuck partway
              u = path_list[index]
              v = path_list[index + 1]

              if (self.G[u][v]['capacity'] < 0):
                od_made = False
                break

            if od_made: # GB BUG FIX #2, continued: only assign flow to edges if the trip can be made (i.e. all edges have non-zero capacity)
              for index in range(0, len(path_list) - 1):
                u = path_list[index]
                v = path_list[index + 1]

                try:
                  if (self.G[u][v]['capacity']>0): # if the edge capacity is greater than 0
                    self.G[u][v]['flow'] += od_flow # assign the flow to the edge
                    t = util.TravelTime(self.G[u][v]['t_0'], self.G[u][v]['capacity']) # GB QUESTION -- does t_0 never get updated???
                    travel_time= t.get_new_travel_time(self.G[u][v]['flow']) # GB BUG FIX #3 -- CHANGE PER JML
                    self.G[u][v]['t_a'] = travel_time #in seconds

                  else:
                    od_made = False # if we couldn't make the trip between node u and node v, change od_made to False
                    break # GB ADDITION -- we should not continue assigning traffic to edges between origin and destination if destination is not reachable from origin!


                except KeyError as e:
                  print('found key error: ', e)
                  pdb.set_trace()

            trips_made += od_made*od_flow # if we manage to make it all the way from origin to destination along the path in path_list, add the od_flow to the total_flow

          except:
            lost_trips += od_flow
            pass

    return self.G, trips_made


def test():
  #create graph info
  G = nx.MultiDiGraph()
  G.add_node(1)
  G.add_node(2)
  G.add_node(3)
  G.add_edge(1,2,capacity_0=1000,capacity=1000,t_0=15,t_a=15,flow=0, distance=10)
  G.add_edge(1,2,capacity_0=3000,capacity=3000,t_0=20,t_a=20,flow=0, distance=10)
  G.add_edge(2,3, capacity_0=0, capacity=0, t_0 = 10, t_a = 10, flow = 0, distance = 10)
  #get od info. This is in format of a dict keyed by od, like demand[sd1][sd2] = 200000.
  demand = {}
  demand[1] = {}
  demand[1][2] = 8000 #divide by 0.053 since that is what we multiply by above
  demand[2] = {}
  demand[2][3] = 4000

  #call ita
  it = ITA(G,demand)
  newG, total_flow, total_demand, lost_flow, od_pairs_lost = it.assign()
  print(newG)
  print('total flow:', total_flow)
  print('total demand:', total_demand)
  print('proportion of demand met:', (total_flow/total_demand))
  for n,nbrsdict in newG.adjacency_iter():
    for nbr,keydict in nbrsdict.items():
      for key,eattr in keydict.items():
        print (n, nbr, eattr['flow'])
  print('should have flow of 3200 and 4800')

if __name__ == '__main__':
  test()
