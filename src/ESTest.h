#include <boost/python.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/shared_ptr.hpp>

#endif

#include <boost/shared_ptr.hpp>

#include <vector>
#include <queue>

#include "NeuralNetwork.h"
#include "Substrate.h"
#include "Innovation.h"
#include "Genes.h"
#include "Assert.h"
#include "PhenotypeBehavior.h"
#include "Random.h"
#include "Genome.h"

namespace NEAT
{

  class ESTest
  {
    Genome testG;
    std::vector<double> test_values;
    std::vector<Connection> test_connections;
    Quadpoint test_point;

    ESTest();

    void runTests();

    bool testCollectValue();

    bool testVariance();

    bool testPruneExpress();

    bool testDivideInitialize();

    bool testCleanNet();

    bool testPhenotypeBuild();

  };
}
