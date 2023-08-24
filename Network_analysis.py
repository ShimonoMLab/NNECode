#!/usr/bin/env python
# coding: utf-8

#pip install bctpy
#pip install python-louvain

import numpy as np
import pandas as pd
import sys
import shutil
import os

import networkx as nx
import matplotlib.pyplot as plt
import community
import bct


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()
    
    
dataIN = []
GnDIN = []
GIN = []
dataIN = pd.read_csv("./{}/pdf.txt".format(sys.argv[1]),header = None,sep=",")

cellcateg  = pd.read_csv("./{}/cell_categ_exc.txt".format(sys.argv[1]),header = None,sep=",")
layercateg = pd.read_csv("./{}/layer_categ.txt".format(sys.argv[1]),header = None,sep=",")
categ = (cellcateg)*8+(7-layercateg)

GnDIN = nx.from_pandas_adjacency(dataIN)
GIN = nx.from_pandas_adjacency(dataIN,create_using = nx.DiGraph())
GnDIN.remove_edges_from(nx.selfloop_edges(GnDIN))
GIN.remove_edges_from(nx.selfloop_edges(GIN))
CoreNumber = nx.core_number(GIN)
CoreNumberPD = pd.DataFrame(CoreNumber,index=[0])
dataIN["CCat"] = categ[0].values.T
dataIN = dataIN.sort_values("CCat",ascending=False,kind='mergesort')
dataIN = dataIN.drop("CCat",axis=1)   # 1 --> 0
dataIN = dataIN.T
dataIN["CCat"] = categ[0].values.T
dataIN = dataIN.sort_values("CCat",ascending=False,kind='mergesort')
dataIN = dataIN.drop("CCat",axis=1)   # 1 --> 0
dataIN = dataIN.iloc[0:100,0:100]
dataIN = dataIN.T
dataIN.index   = range(100)
dataIN.columns = range(100)

GnD = []
G = []
GnD = nx.from_pandas_adjacency(dataIN)
G = nx.from_pandas_adjacency(dataIN,create_using = nx.DiGraph())
GnD.remove_edges_from(nx.selfloop_edges(G))
G.remove_edges_from(nx.selfloop_edges(G))

#  filing the diagonal components with zeros 
for kk in range(0,len(dataIN)):
   dataIN[kk][kk] = 0

Degree = nx.degree_centrality(G)
InDegree = nx.in_degree_centrality(G)
PageRank = nx.pagerank(G)
Betweenness = nx.betweenness_centrality(G)
SubgraphCentrality = nx.subgraph_centrality(GnD)
CoreNumber = nx.core_number(G)
ClusterCoefficients = nx.clustering(G)
Closeness = nx.closeness_centrality(G)
GlobalEfficiency = nx.global_efficiency(GnD)
LE = np.zeros(len(GnD))
for nn1 in range(0,len(GnD)):
    aa = np.zeros(len(GnD))
    for nn2 in range(0,len(GnD)):
        if nn1 != nn2:
            aa[nn2] = nx.efficiency(GnD,nn1,nn2)
    LE[nn1] = np.mean(aa,axis=0)
LocalEfficiency = LE
md = community.best_partition(GnD)
participation = bct.participation_coef(dataIN.values,ci=list(md.values()))

DegreePD = pd.DataFrame(Degree,index=[0])
InDegreePD = pd.DataFrame(InDegree,index=[0]) 
PageRankPD = pd.DataFrame(PageRank,index=[0])
BetweennessPD = pd.DataFrame(Betweenness,index=[0])
SubgraphCentralityPD = pd.DataFrame(SubgraphCentrality,index=[0])
CoreNumberPD = pd.DataFrame(CoreNumber,index=[0])
ClusterCoefficientsPD = pd.DataFrame(ClusterCoefficients,index=[0])
ClosenessPD = pd.DataFrame(Closeness,index=[0])

RF = []
RF = pd.concat([DegreePD,InDegreePD,PageRankPD,BetweennessPD,SubgraphCentralityPD,CoreNumberPD,ClusterCoefficientsPD,ClosenessPD]) #on 2023 Jupy 13

GlobalEfficiencyPD = pd.Series(GlobalEfficiency,index = RF.columns)
LocalEfficiencyPD = pd.Series(LocalEfficiency,index = RF.columns)
participationPD = pd.Series(participation,index = RF.columns)

# our new metric from network embedding analysis
def _our_metrics2(G,nodes=None, weight=None):
    avg1 = {}
    avg2 = {}
    avg3 = {}
    navg1 = {}
    navg2 = {}
    navg3 = {}
    dev1 = {}
    sdev1 = {}
    dev2 = {}
    avg5 = {}
    for n in G.nodes:
        deg = G.degree[n]
        if deg == 0:
            deg = 1
        nbrdeg1 =  nx.single_source_shortest_path_length(G, n, cutoff=0)
        avg1[n] = sum(G.degree[n] for n in nbrdeg1) 
        navg1[n] = sum(1 for n in nbrdeg1)
        nbrdeg2 =  nx.single_source_shortest_path_length(G, n, cutoff=1)
        avg2[n] = sum(G.degree[n] for n in nbrdeg2) 
        navg2[n] = sum(1 for n in nbrdeg2)
        nbrdeg3 =  nx.single_source_shortest_path_length(G, n, cutoff=2)
        avg3[n] = sum(G.degree[n] for n in nbrdeg3) 
        navg3[n] = sum(1 for n in nbrdeg3)
        dev1[n] = navg2[n] - navg1[n]
        dev2[n] = navg3[n] - navg2[n]
        if dev1[n] == 0:
           # dev1[n] = 1 
            dev1[n] = 0.5
        if dev2[n] == 0:
           # dev2[n] = 1
            dev2[n] = 0.5
        sdev1[n] = ((avg2[n] - avg1[n])/float(dev1[n]))
        if sdev1[n] == 0:
           # sdev1[n] = 1
            sdev1[n] = 0.5
        avg5[n] = ((avg3[n] - avg2[n])/float(dev2[n]))/sdev1[n]
    return avg5
        
NewMetric = _our_metrics2(G)
NewMetricPD = pd.Series(NewMetric,index = RF.columns)

#function to calculate new metrics
def _new_metric_threshold(G,steps = 2, tops=0.2, nodes = None, weight=None):
    hubs = {}
    nnodes = {}
    ave5 = {}
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    thres_degree = degree_sequence[int(nx.number_of_nodes(G)*tops)]
    for n in G.nodes:
        hubs[n] = 0
        nnodes[n] = 0
        deg = G.degree[n]
        nbr =  nx.single_source_shortest_path_length(G, n)
        nbrda = [k for k, v in nbr.items() if v == int(steps)]
        if len(nbrda) >= 1:
           for nm in nbrda:
              if G.degree[nm] >= thres_degree:
                 hubs[n] = hubs[n] + 1
              nnodes[n] = nnodes[n] + 1
           ave5[n] = float(hubs[n])/float(nnodes[n])
        if len(nbrda) == 0:
           ave5[n] = 0
    return ave5

NewMetricTH_1 = _new_metric_threshold(G,steps = 1, tops = 0.2)
NewMetricTH_1 = pd.Series(NewMetricTH_1,index = RF.columns)
NewMetricTH_2 = _new_metric_threshold(G,steps = 2, tops = 0.2)
NewMetricTH_2 = pd.Series(NewMetricTH_2,index = RF.columns)
NewMetricTH_3 = _new_metric_threshold(G,steps = 3, tops = 0.2)
NewMetricTH_3 = pd.Series(NewMetricTH_3,index = RF.columns)
NewMetricTH_4 = _new_metric_threshold(G,steps = 4, tops = 0.2)
NewMetricTH_4 = pd.Series(NewMetricTH_4,index = RF.columns)

RF = RF.T
RF = pd.concat([RF,GlobalEfficiencyPD],axis=1)
RF = pd.concat([RF,LocalEfficiencyPD],axis=1)
RF = pd.concat([RF,participationPD],axis=1)
RF = pd.concat([RF,NewMetricPD],axis=1)
RF = pd.concat([RF,NewMetricTH_1],axis=1)
RF = pd.concat([RF,NewMetricTH_2],axis=1)
RF = pd.concat([RF,NewMetricTH_3],axis=1)
RF = pd.concat([RF,NewMetricTH_4],axis=1)

RF = RF.T

# RF.index = ['Degree','PageRank','Betweenness','SubgraphCentrality','CoreNumber','ClusterCoefficients','Closenness','GlobalEfficiency','LocalEfficiency','Participation','NewMetric_Previous','NetwMetric_steps1','NetwMetric_steps2','NetwMetric_steps3','NetwMetric_steps4',"PCA1","PCA2","PCA3","PCA4","PCA5","PCA6"]
RF.index = ['Degree','InDegree','PageRank','Betweenness','SubgraphCentrality','CoreNumber','ClusterCoefficients','Closenness','GlobalEfficiency','LocalEfficiency','Participation','Non-adjacent degree','1st NeighHubRatio','2nd NeighHubRatio','3rd NeighHubRatio','4th NeighHubRatio']
print(RF)

RF.to_csv("./{}/Results_of_Network_analysis_SizeSame2_mod.txt".format(sys.argv[1]),sep="\t")

RF = RF.T
RF.to_csv("./{}/Results_of_Network_analysis_SizeSame2_viz.txt".format(sys.argv[1]),sep="\t")

dataIN.to_csv("./{}/pdf_sorted.txt".format(sys.argv[1]),header = None,sep=",")
Ge = nx.to_edgelist(G)
Ge = pd.DataFrame(Ge)
Ge = Ge.drop([2],axis=1)
Ge.to_csv("./{}/list_pdf_sorted.txt".format(sys.argv[1]),header = None,sep=",")
