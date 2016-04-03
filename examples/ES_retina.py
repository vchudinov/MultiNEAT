#!/usr/bin/python3
import os
import sys
import time
import random as rnd
import subprocess as comm
import cv2
import numpy as np
import pickle as pickle
import MultiNEAT as NEAT
from MultiNEAT import *
import itertools
# GetGenomeList, ZipFitness
#from MultiNEAT.tools import EvaluateGenomeList_Serial

from concurrent.futures import ProcessPoolExecutor, as_completed


params = NEAT.Parameters()
params.PopulationSize = 125;

params.DynamicCompatibility = True;
params.CompatTreshold = 2.0;
params.YoungAgeTreshold = 15;
params.SpeciesMaxStagnation = 100;
params.OldAgeTreshold = 35;
params.MinSpecies = 5;
params.MaxSpecies = 10;
params.RouletteWheelSelection = False;

params.MutateRemLinkProb = 0.02;
params.RecurrentProb = 0;
params.OverallMutationRate = 0.15;
params.MutateAddLinkProb = 0.08;
params.MutateAddNeuronProb = 0.01;
params.MutateWeightsProb = 0.90;
params.MaxWeight = 8.0;
params.WeightMutationMaxPower = 0.2;
params.WeightReplacementMaxPower = 1.0;

params.MutateActivationAProb = 0.0;
params.ActivationAMutationMaxPower = 0.5;
params.MinActivationA = 0.05;
params.MaxActivationA = 6.0;

params.MutateNeuronActivationTypeProb = 0.03;

params.ActivationFunction_SignedSigmoid_Prob = 1.0;
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0;
params.ActivationFunction_Tanh_Prob = 1.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = 1.0;
params.ActivationFunction_UnsignedStep_Prob = 0.0;
params.ActivationFunction_SignedGauss_Prob = 1.0;
params.ActivationFunction_UnsignedGauss_Prob = 0.0;
params.ActivationFunction_Abs_Prob = 0.0;
params.ActivationFunction_SignedSine_Prob = 1.0;
params.ActivationFunction_UnsignedSine_Prob = 0.0;
params.ActivationFunction_Linear_Prob = 1.0;

params.DivisionThreshold = 0.5;
params.VarianceThreshold = 0.03;
params.BandThreshold = 0.3;
params.InitialDepth = 3;
params.MaxDepth = 4;
params.IterationLevel = 1;
params.Leo = False;
params.GeometrySeed = False;
params.LeoSeed = False;
params.LeoThreshold = 0.3;
params.CPPN_Bias = 1.0;
params.Qtree_X = 0.0;
params.Qtree_Y = 0.0;
params.Width = 1.;
params.Height = 1.;
params.Elitism = 0.1;

rng = NEAT.RNG()
rng.TimeSeed()
left_patterns = [
[0., 0., 0., 0.],
[0., 0., 0., 1,],
[0., 1., 0., 1.],
[0., 1., 0., 0.],
[0., 1., 1., 1.],
[0., 0., 1., 0.],
[1., 1., 0., 1.],
[1., 0., 0., 0.]
]

right_patterns = [
[0., 0., 0., 0],
[1., 0., 0.,0.],
[1., 0., 1., 0.],
[0., 0., 1., 0],
[1., 1., 1., 0.],
[0., 1., 0., 0.,],
[1., 0., 1., 1.],
[0., 0., 0., 1]
]
possible_inputs = [list(x) for x in itertools.product([1, 0], repeat = 8)]


substrate = NEAT.Substrate(
        [(-1.0,-1.0, 1.0),(-1, 1, 1),(-0.33, -1, 1),(-0.33, 1, 1),
        (0.33, -1, 1), (0.33, 1, 1),(1, -1, 1),(1, 1, 1),
        (-1, -1, -1)],
        [(-1, -1, 0.75),(-1, 1, 0.75),(-0.33, -1, 0.75),(-0.33, 1, 0.75),
         (0.33, -1, 0.75),(0.33, 1, 0.75), (1, -1, 0.75),(1, 1, 0.75),(-1, 0, 0.5),
         (-0.33, 0, 0.5), (0.33, 0, 0.5), (1, 0, 0.5),(-0.5, 0, 0.25),(0.5, 0, 0.25) ],
        [(0, -1, 0), (0, 1, 0) ]
        )

substrate.m_allow_input_hidden_links = False;
substrate.m_allow_input_output_links = False;
substrate.m_allow_hidden_hidden_links = False;
substrate.m_allow_hidden_output_links = False;
substrate.m_allow_output_hidden_links = False;
substrate.m_allow_output_output_links = False;
substrate.m_allow_looped_hidden_links = False;
substrate.m_allow_looped_output_links = False;

substrate.m_allow_input_hidden_links = True;
substrate.m_allow_input_output_links = False;
substrate.m_allow_hidden_output_links = True;
substrate.m_allow_hidden_hidden_links = False;

substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID;
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID;

substrate.m_with_distance = False;

substrate.m_max_weight_and_bias = 8.0;

def evaluate_retina_and(genome):
    error = 0
    correct = 0.

    try:
        net = NEAT.NeuralNetwork();
        start_time = time.time()
        #genome.BuildHyperNEATPhenotype(net, substrate)
        genome.BuildESHyperNEATPhenotype(net, substrate, params)
        end_time = time.time() - start_time
        left = False
        right = False

        for i in possible_inputs:

            left = i[0:4] in left_patterns
            right = i[4:] in right_patterns
            inp = i[:]
            inp.append(1) #bias

            net.Flush()
            net.Input(inp)
            [net.Activate() for _ in range(5)]
            output = net.Output()

            if (left and right):
                if output[0] > 0.5 and output[1] > 0.5:
                    correct +=1.
                error += abs(1.0 - output[0])
                error += abs(1.0 - output[1])

            elif left:
                if output[0] > 0.5 and output[1] <= 0.5:
                    correct +=1.

                error += abs(1.0 - output[0])
                error += abs(-1.0 - output[1])

            elif right:
                if output[0] <= 0.5 and output[1] > 0.5:
                    correct +=1.

                error += abs(-1.0 - output[0])
                error += abs(1.0 - output[1])

            else:
                if output[0] <= 0.5 and output[1] <= 0.5:
                    correct +=1.
                error += abs(-1.0 - output[0])
                error += abs(-1.0 - output[1])

        return 1000.0 /( 1.0 + error * error)

    except Exception as ex:
        print "nn ",ex
        return 0.0



def getbest(run):
    g = NEAT.Genome(0,
                    substrate.GetMinCPPNInputs(),
                    0,
                    substrate.GetMinCPPNOutputs(),
                    False,
                    NEAT.ActivationFunction.TANH,
                    NEAT.ActivationFunction.TANH,
                    0,
                    params)

    pop = NEAT.Population(g, params, True, 1.0, run)
    for generation in range(10000):
        #Evaluate genomes
        genome_list = NEAT.GetGenomeList(pop)
        print "Start: "
        fitnesses = EvaluateGenomeList_Serial(genome_list, evaluate_retina_and, display=False)
        print "Evaluated"
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

        print('Gen: %d Best: %3.5f' % (generation, max(fitnesses)))

        # Print best fitness
        #print("---------------------------")
        #print("Generation: ", generation)
        #print("max ", max([x.GetLeader().GetFitness() for x in pop.Species]))


        # Visualize best network's Genome

        #if max(fitnesses) > 15.0:
        #    break

        # Epoch
        generations = generation
        pop.Epoch()
        print "---------------------------------"

    return generations


gens = []
for run in range(1):
    gen = getbest(run)
    gens += [gen]
    print('Run:', run, 'Generations to solve XOR:', gen)
avg_gens = sum(gens) / len(gens)

print('All:', gens)
print('Average:', avg_gens)
