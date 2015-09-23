import matplotlib.pyplot as plt
import numpy as np
import MultiNEAT as NEAT
import networkx as nx
import community

params = NEAT.Parameters()
params.DivisionThreshold = 0.5
params.VarianceThreshold = .03
params.BandThreshold = 0.03
params.InitialDepth = 3
params.MaxDepth = 5
params.IterationLevel = 1
params.Leo = True
params.LeoSeed = False
params.GeometrySeed = False
params.LeoThreshold = 0.0
params.CPPN_Bias = -3.0
params.Qtree_X = 0.0
params.Qtree_Y = 0.0
params.Width = 1.0
params.Height = 1.0

substrate = NEAT.Substrate(
        [(-1.0,-1.0, 1.0),(-.33,-1.0,1.0),(0.33,-1.0,1.0),(1.0,-1.0,1.0),
        (-1.0,-1.0,-1.0), (-0.33,-1.0,-1.0),(0.33,-1.0,-1.0),(1.0,-1.0,-1.0),
        (0.0,-1.0,0.0)],
        [(-1.0, -.5, .0),(-.75,-.5,.0),(-0.5,-0.5,.0),(-.25, -.5, .0),
         (.25, -.5, .0), (.5, -.5, 0.), (.75, -.5, 0.0), (1., -.5, 0.),
         (-.5, 0.0,0.0),(-.25, 0.0, 0.0),
         (.25, 0.0, 0.0),(.5, 0, 0),
         (-.25, 0.5, 0),
         (.25, .5, 0)],
        [(0.0,1.0,0.0)] #(-1.,1,0),(1,1,0)]
        )
substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
substrate.m_output_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID

substrate.m_allow_input_hidden_links = True
substrate.m_allow_input_output_links = False
substrate.m_allow_hidden_output_links = True
substrate.m_allow_hidden_hidden_links = True
# when to output a link and max weight
substrate.m_link_threshold = 0.0
substrate.m_max_weight_and_bias = 8.0

def get_neuron_indices(connections):
    indices = []
    for connection in connections:
        if connection.source_neuron_idx not in indices:
            indices.append(connection.source_neuron_idx)
        if connection.target_neuron_idx not in indices:
            indices.append(connection.target_neuron_idx)
    return indices

def parse_network(genome,substrate, params, es = True):
    nn = NEAT.NeuralNetwork()
    if es:
        genome.Build_ES_Phenotype(nn, substrate, params)
    else:
        genome.BuildHyperNEATPhenotype(nn, substrate)
    indices = get_neuron_indices(nn.connections)
    connections = [(c.source_neuron_idx, c.target_neuron_idx) for c in nn.connections]
    #create graph:

    assert len(indices) > 0
    assert len(connections) > 0

    DG=nx.Graph()
    # populate graph
    [ DG.add_node(i) for i in indices]
    DG.add_edges_from(connections)

    #calculate stuff
    part = community.best_partition(DG)
    mod = community.modularity(part,DG)

    return mod


es_nsga_mod = []
for i in range (20):
    g = NEAT.Genome("/home/penguinofdoom/Projects/data/retina/es_nsga_genomes/retina_NSGA_3000_" + str(i) + ".gen")
    es_nsga_mod.append(parse_network(g, substrate,params))

hn_nsga_mod = []
for i in range (20):
    g = NEAT.Genome("/home/penguinofdoom/Projects/data/retina/hn_nsga_genomes/retina_HN_NSGA_2900_" + str(i) + ".gen")
    hn_nsga_mod.append(parse_network(g, substrate,params, False))

es_seeded_mod = []
for i in range (20):
    g = NEAT.Genome("/home/penguinofdoom/Projects/data/retina/es_seeded_retina_genomes/best_" + str(i) + "_3000")
    es_seeded_mod.append(parse_network(g, substrate,params, True))

hn_seeded_mod = []
for i in range (20):
    g = NEAT.Genome("/home/penguinofdoom/Projects/data/retina/hn_seeded_retina/best_" + str(i) + "_3000")
    hn_seeded_mod.append(parse_network(g, substrate,params, False))

hn_control_mod = []
for i in range (20):
    g = NEAT.Genome("/home/penguinofdoom/Projects/data/retina/hn_retina_unseeded/unseeded_hn_3000_" + str(i) + ".gen")
    hn_control_mod.append(parse_network(g, substrate,params, False))

params.Leo = False
es_pure_mod = []
for i in range (15):
    try:
        g = NEAT.Genome("/home/penguinofdoom/Projects/data/new-data/es_true_clean_" + str(i) + "_2800")
        es_pure_mod.append(parse_network(g, substrate,params))
    except AssertionError as e:
        continue

es_nsga_no_leo  = []
for i in range (10):
    print i
    g = NEAT.Genome("/home/penguinofdoom/Projects/data/new-data/es_nsga_no_leo_3000_" + str(i))
    es_nsga_no_leo.append(parse_network(g, substrate,params))

conds = [" ES-HyperNEAT-CCT ", " ES-HyperNEAT\nSeeded ", " HyperNEAT-CCT ", " HyperNEAT\nSeeded ", "HyperNEAT-LEO", "ES-HyperNEAT\nPure", "ES-HyperNEAT-CCT\nNo Leo"]
colors = ['blue', 'green', 'lightblue', 'lightgreen', 'gray', 'orange']

datalist = [es_nsga_mod, es_seeded_mod, hn_nsga_mod, hn_seeded_mod, hn_control_mod, es_pure_mod, es_nsga_no_leo]

means = [np.average(i) for i in datalist]
print means
error = [np.std(i) for i in datalist]

fig, ax1 = plt.subplots(figsize=(8,6))
plt.errorbar([1,2,3,4,5,6, 7], means, yerr= error, fmt='o', elinewidth = 2, markersize = 10, markeredgecolor = 'k',markerfacecolor = 'w', ecolor = 'k')
counter = 0.1

for i in range(len(means)):
    counter +=1
    ax1.text(counter, means[i], "%.4f" %(means[i]))


ax1.set_title("Average Modularity Score On The Retina Task At Generation 3000", fontweight = 'bold', size = 16)
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)
ax1.set_axisbelow(True)
ax1.set_ylabel('Modularity', fontweight = 'bold', size = 16)
ax1.set_xticks([1,2,3,4, 5, 6,7])
ax1.set_xticklabels(conds)
ax1.set_xlim(0.0, 7.5)
ax1.set_ylim(-0.05, 0.3)
ax1.tick_params(axis='both', which='major', labelsize=16)

plt.show()
