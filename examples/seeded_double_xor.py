#!/usr/bin/python
import os
import sys

sys.path.append("/home/penguinofdoom")
sys.path.append("/home/penguinofdoom/Projects")
sys.path.append("/home/penguinofdoom/Projects/Retina")
import itertools
import numpy as np
import MultiNEAT as NEAT
import multiprocessing as mpc
import os.path
import cv2
import Utilities
import traceback
import scipy.stats as ss
import gc
import time
# NEAT parameters

params = NEAT.Parameters()
params.PopulationSize = 150

params.DynamicCompatibility = True
params.MinSpecies = 5
params.MaxSpecies = 15
params.RouletteWheelSelection = False
params.MutateRemLinkProb = 0.0
params.RecurrentProb = 0.0
params.OverallMutationRate = 0.0
params.MutateAddLinkProb = 0.03
params.MutateAddNeuronProb = 0.01
params.MutateWeightsProb = 0.9
params.MaxWeight = 5.0
params.CrossoverRate = 0.5
params.MutateWeightsSevereProb = 0.01
params.TournamentSize = 2;

# Probabilities for a particular activation functiothinner waistn appearance
params.ActivationFunction_SignedSigmoid_Prob = 1
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_TanhCubic_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.0
params.ActivationFunction_UnsignedStep_Prob = 0.0
params.ActivationFunction_SignedGauss_Prob = 1
params.ActivationFunction_UnsignedGauss_Prob = 0.0
params.ActivationFunction_Abs_Prob = 1
params.ActivationFunction_SignedSine_Prob = 1
params.ActivationFunction_UnsignedSine_Prob = 0.0
params.ActivationFunction_Linear_Prob = 1

# ES-HyperNEAT parameters
params.DivisionThreshold = 0.5
params.VarianceThreshold = .03
params.BandThreshold = 0.3
params.InitialDepth = 4
params.MaxDepth = 5
params.IterationLevel = 1
params.Leo = True
params.LeoSeed = True
params.GeometrySeed = True
params.LeoThreshold = 0.0
params.CPPN_Bias = -1.0
params.Qtree_X = 0.0
params.Qtree_Y = 0.0
params.Width = 1.0
params.Height = 1.0

params.Elitism = 0.1
rng = NEAT.RNG()
rng.TimeSeed()

# retina task inputs
possible_inputs = [list(x) for x in itertools.product([1, 0], repeat = 4)]


# Substrates
substrate = NEAT.Substrate(
        [(-1.0,-1.0, 0.0),(-.33,-1.0,0.0),(0.33,-1.0,0.0),(1.0,-1.0,0.0),(0.0,-1.0, 0.0)
        ],
        [(-1.0,0.36, 1.0),(-.33,0.36,1.0),(0.33,0.36,1.0),(1.0,0.36,1.0),
        (-1.0,0.61,-1.0), (-0.33,0.61,-1.0),(0.33,0.61,-1.0),(1.0,0.61,-1.0)],
        [(0.0,1.0,0.0)] #(-1.,1,0),(1,1,0)]
        )
#,(0.0,1.0,0.0)] #
'''
substrate = NEAT.Substrate(
        [(-1.0,-1.0, 1.0),(-1.,1.0, 1.0),(-0.33,-1,1.0),(-0.33,1,1.0),
        (0.33,-1.0,1.0), (0.33,1.,1.0),(1.0,-1.0,1.0),(1.0,1.0,1.0),
        (-1.0,-1.0,-1.0)],
        [],
        [(0.0,-1,0),(0,1,0)]
        )
#'''
substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
substrate.m_output_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID

substrate.m_allow_input_hidden_links = False
substrate.m_allow_input_output_links = False
substrate.m_allow_hidden_hidden_links = False
substrate.m_allow_hidden_output_links = False
substrate.m_allow_output_hidden_links = False
substrate.m_allow_output_output_links = False
substrate.m_allow_looped_hidden_links = False
substrate.m_allow_looped_output_links = False

substrate.m_allow_input_hidden_links = True
substrate.m_allow_input_output_links = False
substrate.m_allow_hidden_output_links = True
substrate.m_allow_hidden_hidden_links = True
# when to output a link and max weight
substrate.m_link_threshold = 0.0
substrate.m_max_weight_and_bias = 8.0
# when to output a link and max weight


# AND task
def evaluate_retina_and(genome):
    error = 0
    correct = 0.

    try:
        net = NEAT.NeuralNetwork();
        start_time = time.time()
        genome.Build_ES_Phenotype(net, substrate, params)
        end_time = time.time() - start_time
        left = False
        right = False

        for i in possible_inputs:

            left = i[0:2] in [[0,1],[1,0]]
            right = i[2:] in [[0,1],[1,0]]
            inp = i[:]
            inp.append(-1)

            net.Flush()
            net.Input(inp)
            [net.Activate() for _ in range(5)]
            output = net.Output()

            if (left and right):
                error += abs(1.0 - output[0])
                if output[0] > 0.:
                    correct +=1.

            else:
                error += abs(-1.0 - output[0])
                if output[0] < 0.:
                    correct +=1.

        return [1000/(1+ error*error), net.GetTotalConnectionLength(), correct/len(possible_inputs), end_time ]

    except Exception as ex:
        print "nn ",ex
        return (0.0, 0.0, 0.0, 0.0)


#OR task
def evaluate_retina_or(genome):
    error = 0
    correct = 0.

    try:
        net = NEAT.NeuralNetwork();
        start_time = time.time()
        genome.Build_ES_Phenotype(net, substrate, params)
        #genome.BuildHyperNEATPhenotype(net, substrate)
        end_time = time.time() - start_time
        left = False
        right = False

        for i in possible_inputs:

            left = i[0:4] in left_patterns
            right = i[4:] in right_patterns
            inp = i[:]
            inp.append(-1)

            net.Flush()
            net.Input(inp)
            [net.Activate() for _ in range(5)]
            output = net.Output()

            if (left or right):
                error += abs(1.0 - output[0])
                if output[0] > 0.:
                    correct +=1.

            else:
                error += abs(-1.0 - output[0])
                if output[0] < 0.:
                    correct +=1.

        return [1000/(1+ error*error), net.GetTotalConnectionLength(), correct/256., end_time ]

    except Exception as ex:
        print "nn ",ex
        return (0.0, 0.0, 0.0, 0.0)


def getbest(run, generations):
    g = NEAT.Genome(0, 7, 1, True, NEAT.ActivationFunction.SIGNED_SIGMOID, NEAT.ActivationFunction.SIGNED_SIGMOID,
            params)
    results = []
    pop = NEAT.Population(g, params, True, 1.0)
    for generation in range(generations):

        genome_list = NEAT.GetGenomeList(pop)
        fitnesses = NEAT.EvaluateGenomeList_Parallel(genome_list, evaluate_retina_and, display = False, cores= 4)

        [genome.SetFitness(fitness[0]) for genome, fitness in zip(genome_list, fitnesses)]
        [genome.SetPerformance(fitness[2]) for genome, fitness in zip(genome_list, fitnesses)]
        [genome.SetLength(fitness[1]) for genome, fitness in zip(genome_list, fitnesses)]

        average_time = np.mean([fitness[3] for fitness in fitnesses])
        max_time = max([fitness[3] for fitness in fitnesses])

        best = pop.GetBestGenome()
        print "---------------------------"
        print "Generation: ", generation
        print "Best ", best.GetFitness(), " Perf: ", best.GetPerformance(), "Len", best.Length
        print "Average time ", average_time, " Longest time: ", max_time


        net = NEAT.NeuralNetwork()
        best.BuildPhenotype(net)
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img += 10
        NEAT.DrawPhenotype(img, (0, 0, 500, 500), net )
        cv2.imshow("CPPN", img)
        net = NEAT.NeuralNetwork()

        best.Build_ES_Phenotype(net, substrate, params)
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        img += 10
        Utilities.DrawPhenotype(img, (0, 0, 500, 500), net, substrate=True )
        cv2.imshow("NN", img)
        cv2.waitKey(1)

        #results.append([run, generation, best.GetFitness(), best.GetPerformance(), best.Length])
        #if generation % 100 == 0:
        #    best.Save("datadump/best_%d_%d" %(run, generation))
        #    Utilities.dump_to_file(results, "datadump/release.csv")
        #    results = []
        #generations = generation
        pop.Epoch()
    return



#runs = 5
for i in range(5):
    getbest(i,10000)
