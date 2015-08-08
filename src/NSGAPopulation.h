#ifndef _NSGAPOPULATION_H
#define _NSGAPOPULATION_H

///////////////////////////////////////////////////////////////////////////////////////////
//    MultiNEAT - Python/C++ NeuroEvolution of Augmenting Topologies Library
//
//    Copyright (C) 2012 Peter Chervenski
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with this program.  If not, see < http://www.gnu.org/licenses/ >.
//
//    Contact info:
//
//    Peter Chervenski < spookey@abv.bg >
//    Shane Ryan < shane.mcdonald.ryan@gmail.com >
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// File:        MSGAPopulation.h
// Description: Definition for the Multiobjective Population class.
///////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <float.h>

#include "Innovation.h"
#include "Genome.h"
#include "PhenotypeBehavior.h"
#include "Genes.h"
#include "Parameters.h"
#include "Random.h"

namespace NEAT
{

//////////////////////////////////////////////
// The Population class
//////////////////////////////////////////////

class NSGAPopulation
{
    /////////////////////
    // Members
    /////////////////////

private:

    // The innovation database
    InnovationDatabase m_InnovationDatabase;

    // next genome ID
    unsigned int m_NextGenomeID;
   // best fitness ever achieved


    // Number of generations since the best fitness changed
    unsigned int m_GensSinceBestFitnessLastChanged;

    // The initial list of genomes


    // Adjusts each species's fitness
    void AdjustFitness();

    // Calculates how many offspring each genome should have
    void CountOffspring();


public:
    double m_BestFitnessEver;

    // Keep a local copy of the best ever genome found in the run
    Genome m_BestGenome;
    Genome m_BestGenomeEver;
    double m_BestFitness;
    std::vector<Genome> m_Genomes;
    std::vector<Genome> TempPop;
    std::vector<double> probabilities;
    // Random number generator
    RNG m_RNG;

    // Evolution parameters
    Parameters m_Parameters;

    // Current generation
    unsigned int m_Generation;

    ////////////////////////////
    // Constructors
    ////////////////////////////

    // Initializes a population from a seed genome G. Then it initializes all weights
    // To small numbers between -R and R.
    // The population size is determined by GlobalParameters.PopulationSize
    NSGAPopulation(const Genome& a_G, const Parameters& a_Parameters, bool a_RandomizeWeights, double a_RandomRange);


    // Loads a population from a file.
    NSGAPopulation(const char* a_FileName);

    ////////////////////////////
    // Destructor
    ////////////////////////////

    // TODO: Major: move all header code into the source file,
    // make as much private members as possible

    ////////////////////////////
    // Methods
    ////////////////////////////

    // Access
    unsigned int NumGenomes() const
    {
    	return m_Genomes.size();
    }

    unsigned int GetGeneration() const { return m_Generation; }
    double GetBestFitnessEver() const { return m_BestFitnessEver; }
    Genome GetBestGenome() const
    {
       // NSGASort();

        return m_Genomes[0];

    }

    void AddIndividual(Genome& a_New);

    // returns an individual randomly selected from the best N%
    Genome GetIndividual();

    // returns a completely random individual
    Genome GetRandomIndividual(RNG& a_RNG) const;

    Genome GetLeader()
    {
      int max = 0.0;

      for (unsigned int i =0; i < m_Genomes.size(); i++)
      {
        // should be called before sorting the population with nsga;
        if (m_Genomes[max].multifitness[0] < m_Genomes[i].multifitness[0])
              max = i;
      }
      return m_Genomes[max];
    };

    unsigned int GetStagnation() const { return m_GensSinceBestFitnessLastChanged; }

    unsigned int GetNextGenomeID() const { return m_NextGenomeID; }
    void IncrementNextGenomeID() { m_NextGenomeID++; }

    Genome& AccessGenomeByIndex(unsigned int const a_idx);

    InnovationDatabase& AccessInnovationDatabase() { return m_InnovationDatabase; }

    // Sorts each species's genomes by fitness
    void NSGASort();

    // Performs one generation and reproduces the genomes
    void Epoch();

    // Saves the whole population to a file
    void Save(const char* a_FileName);

    void Reproduce(std::vector<Genome> &tempPop);

    void MutateGenome( bool t_baby_is_clone, Genome &t_baby);

    void CalculateAverageFitness();
//
    void SetProbabilities(py::list probs)
    { // Always use the diversity objective;
      probabilities.clear();
      probabilities.push_back(1);

      for(int j=0; j<py::len(probs); j++)
      {
        probabilities.push_back(py::extract<double>(probs[j]));
      }
    }
    /*
    void SetProbabilities(std::vector <double> probs)
    {
      probabilities = probs;
    }*/
    /////////////////////
    // NEW STUFF
      //////////////////////
    // Real-Time methods

    // Estimates the estimated average fitness for all species
    void EstimateAllAverages();

    // Reproduce the population champ only
    Genome ReproduceChamp();

    // Removes worst member of the whole population that has been around for a minimum amount of time
    // returns the genome that was just deleted (may be useful)
    Genome RemoveWorstIndividual();

    // The main reaitime tick. Analog to Epoch(). Replaces the worst evaluated individual with a new one.
    // Returns a pointer to the new baby.
    // and copies the genome that was deleted to a_geleted_genome
  //  Genome* Tick(Genome& a_deleted_genome);

    // Takes an individual and puts it in its apropriate species
    // Useful in realtime when the compatibility treshold changes
    void ReassignSpecies(unsigned int a_genome_idx);

    unsigned int m_NumEvaluations;

    ///////////////////////////////
    // Novelty search

    // A pointer to the archive of PhenotypeBehaviors
    // Not necessary to contain derived custom classes.
    std::vector< PhenotypeBehavior >* m_BehaviorArchive;

    // Call this function to allocate memory for your custom
    // behaviors. This initializes everything.
    void InitPhenotypeBehaviorData(std::vector< PhenotypeBehavior >* a_population,
                                   std::vector< PhenotypeBehavior >* a_archive);

    // This is the main method performing novelty search.
    // Performs one reproduction and assigns novelty scores
    // based on the current population and the archive.
    // If a successful behavior was encountered, returns true
    // and the genome a_SuccessfulGenome is overwritten with the
    // genome generating the successful behavior
    bool NoveltySearchTick(Genome& a_SuccessfulGenome);

    double ComputeSparseness(Genome& genome);

    // counters for archive stagnation
    unsigned int m_GensSinceLastArchiving;
    unsigned int m_QuickAddCounter;
    void PrimaryRanking(std::vector<std::vector<Genome*> > &fronts);
    void SecondaryRanking(std::vector<std::vector<Genome*> >& fronts);
    void AssignDistance(std::vector<std::vector<Genome*> > &fronts);
    void GenomicDiversity();
  //  bool CrowdComparison(Genome ls, Genome rs);
    bool Dominate(Genome ls, Genome rs);
    bool StochasticDominate(Genome ls, Genome rs);
    bool StochasticCrowdComparison(Genome ls, Genome rs);

    double mepsd(Genome* ls, Genome* rs);
    void Sort();
    void quickSort(std::vector<Genome*>& front, int left, int right, int index);
void RemoveIndividual(unsigned int a_idx)
{
    ASSERT(a_idx < m_Genomes.size());
    m_Genomes.erase(m_Genomes.begin() + a_idx);
}


};

} // namespace NEAT

#endif
