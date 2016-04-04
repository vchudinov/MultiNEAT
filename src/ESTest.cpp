#include <algorithm>
#include <fstream>
#include <queue>
#include <math.h>
#include <utility>
#include <boost/shared_ptr.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include "Genome.h"
#include "Random.h"
#include "Utils.h"
#include "Parameters.h"
#include "Assert.h"

namespace NEAT
{

class ESTest {
  ESTest();

  void runTests()
  {
    ASSERT(testCollectValue());
    cout << "Passed Value Collection" << endl;
    ASSERT(testVariance());
    cout << "Passed Variance Test" << endl;
    ASSERT(testDivideInitialize());
    cout << "Passed Division Initialization" << endl;
    ASSERT(testPruneExpress());
    cout << "Passed Prune Express";
    ASSERT(testCleanNet());
    cout << "Passed Network Cleaning" << endl;
    ASSERT(testPhenotypeBuild());
    cout << "Passed phenotype Build";

    cout << "Passed all Tests" << endl;
 }



  bool testCollectValue()
  {
    return false;
  }

  bool testVariance()
  {
    return false;
  }

  bool testPruneExpress()
  {
    return false;
  }

  bool testDivideInitialize()
  {
    return false;
  }

  bool testCleanNet()
  {
    return false;
  }

  bool testPhenotypeBuild()
  {
    return false;
  }
};

}
