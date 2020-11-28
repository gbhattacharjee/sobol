#Author: Mahalia Miller
#Date: Jan. 21, 2012, i.e. date of Obama's second inauguration 

#import networkx as nx
# from __future__ import division
import pdb
import math
class TravelTime:
  alpha = 0.15
  beta = 4
  def __init__(self, t_0, cap_0):
    self.t_0 = t_0
    self.cap_0 = cap_0

  def get_new_travel_time(self, flow):
    alpha = 0.15 # 0.15 in recent paper per JML
    beta = 4 # 2.97 in recent paper per JML
  
    return self.t_0*(1 + alpha*(flow/float(self.cap_0))**beta)  

def clean_up_graph(G):
  #takes a graph and makes all flows 0 and makes t_a equal to t_0
  is_multi = G.is_multigraph()
  if is_multi == True:
    for n,nbrsdict in G.adjacency_iter():
      for nbr,keydict in nbrsdict.items():
        for key,eattr in keydict.items():
          eattr['flow'] = 0
          eattr['t_a'] = eattr['t_0']
          eattr['capacity'] = eattr['capacity_0']
          eattr['distance'] = eattr['distance_0']
  else:
    for n,nbrsdict in G.adjacency_iter():
      for key,eattr in nbrsdict.items():
        eattr['flow'] = 0
        eattr['t_a'] = eattr['t_0']
        eattr['capacity'] = eattr['capacity_0']
        eattr['distance'] = eattr['distance_0']
  return G

def find_travel_time(G):
  #G is a networkx graph. returns the cumulative travel time of all drivers.
  travel_time = 0
  is_multi = G.is_multigraph()
  if is_multi == True:
    for n,nbrsdict in G.adjacency_iter():
      for nbr,keydict in nbrsdict.items():
        for key,eattr in keydict.items():
          if eattr['flow'] > 0:
            travel_time += eattr['flow']*eattr['t_a']
            print eattr['t_a']

            # if math.isinf(travel_time):
            #   pdb.set_trace()
  else:
    for n,nbrsdict in G.adjacency_iter():
      for key,eattr in nbrsdict.items():
        if eattr['flow'] > 0:
          travel_time += eattr['flow'] * eattr['t_a']
          # if math.isinf(travel_time):
          #   pdb.set_trace()

  return travel_time

def find_travel_time_sf(G):
  #G is a networkx graph. returns the cumulative travel time of all drivers.
  travel_time = 0
  is_multi = G.is_multigraph()
  if is_multi == True:
    for n,nbrsdict in G.adjacency_iter():
      for nbr,keydict in nbrsdict.items():
        for key,eattr in keydict.items():
          if eattr['flow'] > 0: #TODO: add a condition checking that the node is in sf
            travel_time += eattr['flow']*eattr['t_a']
            # if math.isinf(travel_time):
            #   pdb.set_trace()
  else:
    for n,nbrsdict in G.adjacency_iter():
      for key,eattr in nbrsdict.items():
        if eattr['flow'] > 0: #TODO: add a condition checking that the node is in sf
          travel_time += eattr['flow'] * eattr['t_a']
          # if math.isinf(travel_time):
          #   pdb.set_trace()
  return travel_time


def find_vmt(G):
  #G is a networkx graph. returns the cumulative vehicles miles traveled of all drivers.
  vmt = 0
  is_multi = G.is_multigraph()
  if is_multi == True:
    for n,nbrsdict in G.adjacency_iter():
      for nbr,keydict in nbrsdict.items():
        for key,eattr in keydict.items():
          vmt += eattr['flow']*eattr['distance']
  else:
    for n,nbrsdict in G.adjacency_iter():
      for key,eattr in nbrsdict.items():
        try:
          vmt += eattr['flow']*eattr['distance']
        except:
          pdb.set_trace()
  return vmt

def compute_delay(travel_time, undamaged_travel_time = None):
  if undamaged_travel_time is not None:
    return travel_time - undamaged_travel_time
  else:
    return 0

def write_list(filename, the_list):
  #writes a list to file
  with open(filename, 'wb') as f:
    for item in the_list:
      f.write("%s\n" % item)

def write_2dlist(filename, the_list):
  #writes a list to file
  with open(filename, 'wb') as f:
    for item in the_list:
      for sub_item in item:
        f.write("%s, " % sub_item)
      f.write("\n")

def read_list(filename, skipheader=False):
  #returns list of lists where each inner list is a value
  the_list = []
  with open(filename,'rU') as f:
    read_data = f.read().splitlines()
    if skipheader == True:
      read_data = read_data[1:]
    for row in read_data: #[1:]:
      the_list.append(row)
  return the_list #list 

def read_2dlist(filename, delimiter=',', skipheader=False):
  #returns list of lists where each inner list is a row
  the_list = []
  with open(filename,'rU') as f:
    read_data = f.read().splitlines()
    if skipheader == True:
      read_data = read_data[1:]
    for row in read_data: #[1:]:
      tokens = row.split(delimiter)
      the_list.append(tokens)
  return the_list #list of lists

if __name__ == '__main__':
  import networkx as nx
  G = nx.MultiDiGraph()
  G.add_node(1)
  G.add_node(2)
  G.add_edge(1,2,capacity_0=1000,capacity=1000,t_0=15,t_a=15,flow=0, distance=10)
  G.add_edge(1,2,capacity_0=3000,capacity=3000,t_0=20,t_a=20,flow=0, distance=10)
  print 'should be 0: ', find_travel_time(G)
  G[1][2][0]['flow']=1000
  print 'should be 15000: ', find_travel_time(G)
  print 'should be 10000: ', find_vmt(G)
#
