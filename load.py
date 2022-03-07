import pickle 
import random
import string
import numpy as np
import quantities as pq
import neo
import codecs
import seaborn as sns
from quantities import units
from pint import UnitRegistry
from neo.io.pickleio import PickleIO
from neo.core import (Block, Segment)
from neo import io
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd
import gzip
import csv
from numpy.core.fromnumeric import argmax, argmin, ndim, nonzero, shape, size
import elephant.unitary_event_analysis as ue
import re
r = io.PickleIO( filename = 'unsupervised_STDP_pose_14_trial_15_nest_np1_20211217-024656.pkl')
blck = r.read_block()
spike=blck.segments[0].spiketrains
#print(spike)
print(size(spike))
# s= np.array(spike, dtype='object')
# z= s.reshape(128,128)
# # #print(z)
# #d= s/1*u.ms
# #spiketrains1 = np.vstack((spike))
# # UE = ue.jointJ_window_analysis(
# #     spiketrains1)
# unit="ms"
# # plot_ue(spiketrains1, UE, significance_level=0.05)
# # plt.show()
# z = z.astype(np.str)
# #def Heatmapping (spiketrain, dimensionX, dimensionY):
FlatSpikes = np.reshape(spike, (12, 12))
for i in range (FlatSpikes.shape[0]):
    for j in range (FlatSpikes.shape[1]):
        FlatSpikes[i,j] = len(FlatSpikes[i,j])
FlatSpikes = np.array(FlatSpikes[:, :]).astype(np.float)
flat=np.vstack(FlatSpikes[:, :]).astype(np.float)
pd.DataFrame(FlatSpikes).to_csv("/home/nest/project/test/flatspike_10.csv")
# ax = sns.heatmap(FlatSpikes,cmap='Blues')
# plt.savefig('1')
nx=sns.heatmap(flat,cmap='Blues')
plt.savefig('2')  #  return (FlatSpikes)
#print(a)
#print(d)
# #print(spike)
#pd.DataFrame(z).to_csv("/home/nest/project/test/spike5.csv")
# #print(z)
# with open('spike_new.csv', 'rb') as fd:
#     gzip_fd = gzip.GzipFile(fileobj=fd)
#     destinations = pd.read_csv(gzip_fd)
# with codecs.open("destinations.csv", "r",encoding='utf-8', errors='ignore') as file_dat:
#      destinations = pd.read_csv(file_data))
#ss= pd.read_csv("/home/nest/project/test/spike_new.csv",encoding= 'unicode_escape')
#zz= destinations.astype(np.float)
# zz= ss.to_numpy()
# ff= zz.astype(np.float)
#data = io.PickleIO(filename='unsupervised_STDP_pose_0_trial_0_nest_np1_20211111-124351.pkl')
# infile = open('unsupervised_STDP_pose_0_trial_0_nest_np1_20211111-124351.pkl','rb')
# data = pickle.load(infile)
# plt.hist(spike)
# plt.show
#print(data)
