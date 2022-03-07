# TODO: Understand each line of the code
# TODO: Connect the code to your data folder by setting the "DATAPATH" environment variable and changing values of dataset_path and metadata_path
# TODO: Play around with parameters to see how they affect the output
# TODO: Display the locations of the neurons of each layer in space
# TODO: Change connection from input to hidden layer from alltoall to  
# TODO: Display the outputs of the neurons of each layer as spike raster plots
# TODO: Display the outputs of the neurons of each layer as maps of spike counts

# encoding: utf-8
import os, pickle, json
import numpy as np
import pyNN.nest as sim
from pyNN.utility import normalized_filename
from pyNN.utility.plotting import Figure, Panel
from pyNN.parameters import Sequence
from pyNN.random import RandomDistribution as rnd, NumpyRNG

############################ FUNCTIONS ##################################

# Remove duplicate timestamps and sort the raw data
def remove_duplicates(data):
    new_data = []
    for i in data:
        new_data.append(np.sort(np.unique(i)))      
    return new_data

# Format data as Sequence to import into pyNN.nest
def sequence_data (pre_data, no_of_pixels):
    inputarray=[None] * (no_of_pixels)
    for x in range (0, no_of_pixels):
        if len(pre_data[x]) > 0:
            inputarray[x] = Sequence(pre_data[x])
        else:
            inputarray[x] = Sequence([])
    return inputarray
 
############################ PARAMETERS #################################

sensor = 'neuroTac_DAVIS240' # sensor used. Options: 'neuroTac_DAVIS240','neuroTac_eDVS','neuroTac_DVXplorer'
simulator = 'nest'

# Neuron type setup
cell_parameters = {
    "tau_m": 10.0,       # cell membrane time constant -- (ms)
    "tau_refrac": 0.5,    # refractory period duration -- (ms)
    "cm": 1.0,           # membrane capacitance -- (nF)
    "tau_syn_E": 1.0,      # rise time of the excitatory synaptic function -- (ms)
    "tau_syn_I": 1.0,      # rise time of the inhibitory synaptic function -- (ms)
    "i_offset": 0.0,     # Offset current.
    "v_thresh": -50.0,   # spike threshold -- (mV)
    "v_reset": -65.0,    # reset potential post-spike -- (mV)
    "v_rest": -65.0,     # resting membrane potential -- (mV)
}
neuron_type = sim.IF_curr_exp(**cell_parameters)
weights = 0.2  # synaptic weight (µS)

# Simulation setup
dt         = 1.0           # (ms)
syn_delay  = 1.0           # (ms)
simtime    = 1000          # (ms)
sim.setup(timestep=dt, min_delay=syn_delay, max_delay=syn_delay)

if(sensor == 'neuroTac_DAVIS240'):
    n_pixels = 240*180
else:
    print('Sensor type not recognized')

######################### LOAD METADATA #############################

data_folder = 'collect_taps_orientation'
metadata_path = os.path.join(os.environ['DATAPATH'], 'NeuroTac_DAVIS240', 'ABB_edge_orientation', data_folder,'meta.json')
with open(metadata_path) as f:
  meta = json.load(f)

n_poses = len(meta['obj_poses'])
n_trials = meta['n_trials']

######################### BUILD NETWORK #############################

# Input layer structure
structure_input = sim.space.Grid2D(aspect_ratio  = 4/3, dx = 1.0, dy = 1.0, fill_order = 'sequential')  
structure_input.generate_positions(n_pixels)

# Hidden layer (filter)
reduction_factor = 10 # size of the receptive fields of the hidden layer (number of pixels on each side)
structure_hidden = sim.space.Grid2D(aspect_ratio  = 4/3, dx = reduction_factor, dy = reduction_factor, fill_order = 'sequential')  
structure_hidden.generate_positions(n_pixels//(reduction_factor*reduction_factor))
layer_hidden = sim.Population(n_pixels//(reduction_factor*reduction_factor), neuron_type, structure = structure_hidden, label = 'layer_hidden')  
layer_hidden.record('spikes')

# Connections
connection_type_input_hidden = sim.DistanceDependentProbabilityConnector("d<"+str(reduction_factor))

########################### LOAD DATA ###############################

# Loop over poses/trials
n_poses = 1
n_trials = 2

for pose_idx in range (n_poses):
    for trial_idx in range (n_trials):
        print('Running simulation_pose_'+str(pose_idx)+'_trial_'+str(trial_idx))
        dataset_filename = 'taps_orientation_trial_' +str(trial_idx) +'_pose_'  + str(pose_idx) + '.pickle'
        dataset_path = os.path.join(os.environ['DATAPATH'], 'NeuroTac_DAVIS240', 'ABB_edge_orientation', data_folder,dataset_filename)
        data_in = open(dataset_path, "rb")
        loaded_data = pickle.load(data_in)
        pre_data = loaded_data.flatten('C') # Load and flatten data to 1D array

        n_pixels = pre_data.size
        simtime = max(max(pre_data))

        pre_data = remove_duplicates(pre_data)
        input_sequence = sequence_data(pre_data, n_pixels)      # Convert 1D array to a pyNN Sequence class

        # Input layer
        layer_input = sim.Population(n_pixels, sim.SpikeSourceArray (spike_times = input_sequence), structure = structure_input, label = 'layer_input')                                            
        layer_input.record('spikes')
        # connection_input_hidden = sim.Projection(layer_input, layer_hidden, connection_type_input_hidden)
        # connection_input_hidden.setWeights(weights)

        # Run simulation
        sim.run(simtime)

######################### SAVE RESULTS #############################

        print('Saving results')

        filename = normalized_filename("Results", "unsupervised_STDP_pose_" + str(pose_idx) +"_trial_" +str(trial_idx), "pkl",
                                    simulator, sim.num_processes())
        layer_input.write_data(filename, annotations={'script_name': __file__})

        print("Mean firing rate: ", layer_input.mean_spike_count() * 1000.0 / simtime, "Hz")

        data_input = layer_input.get_data().segments[0]
        data_hidden = layer_hidden.get_data().segments[0]

        figure_filename = filename.replace("pkl", "png")

        Figure(
            # raster plot of the presynaptic neuron spike times
            Panel(data_hidden.spiketrains,
                    yticks=True, markersize=0.2, xlim=(0, simtime)),
            # # membrane potential of the postsynaptic neuron
            # Panel(data_input.filter(name='v')[0],
            #         ylabel="Membrane potential (mV)",
            #         data_labels=[p2.label], yticks=True, xlim=(0, simtime)),
            title="Basic network",
            annotations="Simulated with %s" % simulator
        ).save(figure_filename)

        print(figure_filename)

        weights = connection_input_hidden.getWeights()

        sim.end()
        sim.reset()