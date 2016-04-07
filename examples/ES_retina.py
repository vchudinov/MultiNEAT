#!/usr/bin/python3
import os
import sys
import time
import random as rnd
import subprocess as comm
#import cv2
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
params.CompatTreshold = 2.5;
params.YoungAgeTreshold = 15;
params.SpeciesMaxStagnation = 100;
params.OldAgeTreshold = 35;
params.MinSpecies = 5;
params.MaxSpecies = 10;
params.RouletteWheelSelection = False;

params.MutateRemLinkProb = 0.02;
params.RecurrentProb = 0;
params.OverallMutationRate = 0.10;
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

params.MutateNeuronActivationTypeProb = 0.0;

params.ActivationFunction_SignedSigmoid_Prob = .0;
params.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
params.ActivationFunction_Tanh_Prob = 0.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = .0;
params.ActivationFunction_UnsignedStep_Prob = 0.0;
params.ActivationFunction_SignedGauss_Prob = .0;
params.ActivationFunction_UnsignedGauss_Prob = 1.0;
params.ActivationFunction_Abs_Prob = .0;
params.ActivationFunction_SignedSine_Prob = 1.0;
params.ActivationFunction_UnsignedSine_Prob = 1.0;
params.ActivationFunction_Linear_Prob = 1.0;

params.DivisionThreshold = 0.5;
params.VarianceThreshold = 0.3;
params.BandThreshold = 0.3;
params.InitialDepth = 3;
params.MaxDepth = 6;
params.IterationLevel = 1;
params.Leo = False;
params.GeometrySeed = False;
params.LeoSeed = False;
params.LeoThreshold = 0.3
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



        #(-0.33, -1, 1),
        #(-0.33, 1, 1),
        #(0.33, -1, 1),
        #(0.33, 1, 1),

substrate = NEAT.Substrate(
        [(-1.0,-1.0, 1.0),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, -1, -1)],
        [(-1, -1, 0.75),(-1, 1, 0.75),(-0.33, -1, 0.75),(-0.33, 1, 0.75),
         (0.33, -1, 0.75),(0.33, 1, 0.75), (1, -1, 0.75),(1, 1, 0.75),(-1, 0, 0.5),
         (-0.33, 0, 0.5), (0.33, 0, 0.5), (1, 0, 0.5),(-0.5, 0, 0.25),(0.5, 0, 0.25) ],
        [(0, -1, 0), (0, 1, 0) ]
        )

substrate.m_allow_input_hidden_links = False;
substrate.m_allow_input_output_links = False;
substrate.m_allow_hidden_hidden_links = True;
substrate.m_allow_hidden_output_links = True;
substrate.m_allow_output_hidden_links = False;
substrate.m_allow_output_output_links = False;
substrate.m_allow_looped_hidden_links = False;
substrate.m_allow_looped_output_links = False;

substrate.m_allow_input_hidden_links = True;
substrate.m_allow_input_output_links = False;
substrate.m_allow_hidden_output_links = True;
substrate.m_allow_hidden_hidden_links = False;

substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.TANH;
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID;

substrate.m_with_distance = False;

substrate.m_max_weight_and_bias = 5.0;

def evaluate_retina_and(genome):
    possible_inputs = [list(x) for x in itertools.product([1, 0], repeat = 8)]
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
            [net.Activate() for _ in range(4)]
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
                error += abs(output[1])

            elif right:
                if output[0] <= 0.5 and output[1] > 0.5:
                    correct +=1.

                error += abs(output[0])
                error += abs(1.0 - output[1])

            else:
                if output[0] <= 0.5 and output[1] <= 0.5:
                    correct +=1.
                error += abs(output[0])
                error += abs(output[1])

        return 1000.0 /( 1.0 + error * error)

    except Exception as ex:
        print "nn ",ex
        return 0.0

def evaluate_double_xor(genome):

    pos_inputs = [list(x) for x in itertools.product([1, 0], repeat = 4)]

    try:

        net = NEAT.NeuralNetwork();
        start_time = time.time()
        #genome.BuildHyperNEATPhenotype(net, substrate)
        genome.BuildESHyperNEATPhenotype(net, substrate, params)
        left = False
        right = False
        correct = 0.0
        error = 0.0
        for i in pos_inputs:

            left = (1 in i[0:2])  and (0 in i[0:2])
            right = (1 in i[2:]) and (0 in i[2:])
            inp = i[:]
            inp.append(1.0)

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
                error += abs(output[1])

            elif right:
                if output[0] <= 0.5 and output[1] > 0.5:
                    correct +=1.

                error += abs(output[0])
                error += abs(1.0 - output[1])

            else:
                if output[0] <= 0.5 and output[1] <= 0.5:
                    correct +=1.

                error += abs(output[0])
                error += abs(output[1])

        #return 1000.0/(1.0 + error*error)
        return correct / (len(pos_inputs) + 0.0)
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

    #g = NEAT.Genome(0,
    #                substrate.GetMinCPPNInputs(),
    #                substrate.GetMinCPPNOutputs(),
    #                False,
    #                NEAT.ActivationFunction.TANH,
    #                NEAT.ActivationFunction.TANH,
    #                params)

    pop = NEAT.Population(g, params, True, 1.0, run)
    for generation in range(2500):
        #Evaluate genomes
        genome_list = NEAT.GetGenomeList(pop)
        fitnesses = EvaluateGenomeList_Serial(genome_list, evaluate_double_xor, display=False)
        #fitnesses = EvaluateGenomeList_Serial(genome_list, evaluate_retina_and, display=False)
        [genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

        print('Gen: %d Best: %3.5f' % (generation, max(fitnesses)))
        if max(fitnesses) > 0.98:
            break
        generations = generation
        pop.Epoch()


    return generations


gens = []
for run in range(10):
    gen = getbest(run)
    gens += [gen]
    print('Run:', run, 'Generations to solve XOR:', gen)

avg_gens = sum(gens) / len(gens)

print('All:', gens)
print('Average:', avg_gens)
