#include "MetaHeuristics.h"
#include "SearchAlgorithms.h"
#include "../Utils/Vector.h"
#include "../MiscAlgs/Misc.h"
#include "TSP.h"
#include "../ExternalMemoryAlgorithms/CSV.h"
using namespace igmdk;

void TSPCompete()
{
    DEBUG("TSPCompete");
    Vector<Vector<string> > matrix;
    Vector<string> row(2);
    int n = 100;
    DEBUG(n);
    row[0] = "Problem Size";
    row[1] = toString(n);
    matrix.append(row);
    row[0] = "Expected Solution";
    double expected = sqrt(n/2.0);
    DEBUG("expect");
    DEBUG(expected);
    row[1] = toString(sqrt(n/2.0));
    matrix.append(row);
    Vector<int> initialOrder = GlobalRNG().randomCombination(n, n);
    TSPRandomInstance instance(n);

    DEBUG("random");
    row[0] = "Random";
    double score = instance(initialOrder);
    DEBUG(score);
    row[1] = toString(score);
    matrix.append(row);
    DEBUG("LS Reversals");
    row[0] = "LS Reversals";
    score = instance(solveTSPLocalSearch<PermutationProblemReverseMove<TSPRandomInstance> >(
        instance, initialOrder, 10000000, 1000));
    DEBUG(score);
    row[1] = toString(score);
    matrix.append(row);
    DEBUG("SA Reversals");
    row[0] = "SA Reversals";
    score = instance(solveTSPSimulatedAnnealing<PermutationProblemReverseMove<TSPRandomInstance> >(
        instance, initialOrder, 10000, 0.9999, 10000000));
    DEBUG(score);
    row[1] = toString(score);
    matrix.append(row);
    DEBUG("LS Swaps");
    row[0] = "LS Swaps";
    score = instance(solveTSPLocalSearch<PermutationProblemSwapMove<TSPRandomInstance> >(
        instance, initialOrder, 10000000, 1000));
    DEBUG(score);
    row[1] = toString(score);
    matrix.append(row);
    DEBUG("SA Swaps");
    row[0] = "SA Swaps";
    score = instance(solveTSPSimulatedAnnealing<PermutationProblemSwapMove<TSPRandomInstance> >(
        instance, initialOrder, 10000, 0.9999, 10000000));
    DEBUG(score);
    row[1] = toString(score);
    matrix.append(row);
    DEBUG("ILS Swaps");
    row[0] = "ILS Swaps";
    score = instance(solveTSPIteratedLocalSearch(
        instance, initialOrder, 100000, 1000, 100));
    DEBUG(score);
    row[1] = toString(score);
    matrix.append(row);
    DEBUG("B&B");
    row[0] = "B&B";
    score = instance(solveTSPBranchAndBound(instance, initialOrder, 1000000));
    DEBUG(score);
    row[1] = toString(score);
    matrix.append(row);

    //also include RBFS after get good bounds and maybe A* to see if safe
    createCSV(matrix, "TSP100RandResults.csv");
}

void testAStarTSP()
{
    for(int n = 10; n <= 40; n += 5)
    {
        DEBUG(n);
        double expected = sqrt(n/2.0);
        DEBUG(expected);
        Vector<int> initialOrder = GlobalRNG().randomCombination(n, n);
        TSPRandomInstance instance(n);
        DEBUG("random");
        double score = instance(initialOrder);
        DEBUG(score);
        pair<Vector<int>, bool> result = solveTSPAStar(instance, 1000000);
        DEBUG(result.second);
        if(result.second)
        {
            DEBUG("A* solution");
            result.first.debug();
            score = instance(result.first);
            DEBUG(score);
        }
    }
}

void testRBFSTSP()
{
    for(int n = 10; n <= 30; n += 5)
    {
        DEBUG(n);
        double expected = sqrt(n/2.0);
        DEBUG(expected);
        Vector<int> initialOrder = GlobalRNG().randomCombination(n, n);
        TSPRandomInstance instance(n);
        DEBUG("random");
        double score = instance(initialOrder);
        DEBUG(score);
        pair<Vector<int>, bool> result = solveTSPRBFS(instance, 10000000);
        DEBUG(result.second);
        if(result.second)
        {
            DEBUG("RFBS solution");
            result.first.debug();
            score = instance(result.first);
            DEBUG(score);
        }
    }
}


void SmallTSPCompete()
{
    DEBUG("SmallTSPCompete");
    int n = 20;
    DEBUG(n);
    double expected = sqrt(n/2.0);
    DEBUG(expected);
    Vector<int> initialOrder = GlobalRNG().randomCombination(n, n);
    TSPRandomInstance instance(n);

    DEBUG("random");
    double score = instance(initialOrder);
    DEBUG(score);
    DEBUG("LS Reversals");
    score = instance(solveTSPLocalSearch<PermutationProblemReverseMove<TSPRandomInstance> >(
        instance, initialOrder, 10000000, 1000));
    DEBUG(score);
    DEBUG("SA Reversals");
    Vector<int> saSolution = solveTSPSimulatedAnnealing<PermutationProblemReverseMove<TSPRandomInstance> >(
        instance, initialOrder, 10000, 0.9999, 10000000);
    score = instance(saSolution);
    DEBUG(score);
    DEBUG("SA solution");
    saSolution.debug();
    DEBUG("LS Swaps");
    score = instance(solveTSPLocalSearch<PermutationProblemSwapMove<TSPRandomInstance> >(
        instance, initialOrder, 10000000, 1000));
    DEBUG(score);
    DEBUG("SA Swaps");
    score = instance(solveTSPSimulatedAnnealing<PermutationProblemSwapMove<TSPRandomInstance> >(
        instance, initialOrder, 10000, 0.9999, 10000000));
    DEBUG(score);
    DEBUG("ILS Swaps");
    score = instance(solveTSPIteratedLocalSearch(
        instance, initialOrder, 100000, 1000, 100));
    DEBUG("B&B");
    Vector<int> bbSolution = solveTSPBranchAndBound(instance, initialOrder, 1000000);
    score = instance(bbSolution);
    DEBUG(score);
    DEBUG("BB solution");
    bbSolution.debug();
    DEBUG("AStar");
    pair<Vector<int>, bool> result = solveTSPAStar(instance, 1000000);
    DEBUG(result.second);
    if(result.second)
    {
        DEBUG("A* solution");
        result.first.debug();
        score = instance(result.first);
        DEBUG(score);
    }
    DEBUG("RBFS");
    result = solveTSPRBFS(instance, 10000000);
    DEBUG(result.second);
    if(result.second)
    {
        DEBUG("RBFS solution");
        result.first.debug();
        score = instance(result.first);
        DEBUG(score);
    }
}

struct GraphProblem
{
    typedef int STATE_ID;
    static int NULL_STATE(){return -1;}
    typedef EHash<BUHash> HASHER;
    GraphAA<double> graph;
    int from, to;
    int start()const{return from;}
    bool isGoal(int i)const{return i == to;}
    Vector<int> nextStates(int j)const
    {
        Vector<int> result;
        for(GraphAA<double>::AdjacencyIterator i = graph.begin(j);
            i != graph.end(j); ++i) result.append(i.to());
        return result;
    }
    double remainderLowerBound(int i)const{return 0;}
    double distance(int k, int j)const
    {
        for(GraphAA<double>::AdjacencyIterator i = graph.begin(j);
            i != graph.end(j); ++i) if(i.to() == k) return i.data();
        return 0;
    }
};

void testAStar()
{
    typedef GraphAA<double> G;
    G sp;
    GraphProblem Gp;
	for(int i = 0; i < 6; ++i)
	{
		Gp.graph.addVertex();
	}
	Gp.graph.addEdge(0,1,6);
	Gp.graph.addEdge(0,2,8);
	Gp.graph.addEdge(0,3,18);
	Gp.graph.addEdge(1,4,11);
	Gp.graph.addEdge(2,3,9);
	Gp.graph.addEdge(4,5,3);
	Gp.graph.addEdge(5,2,7);
	Gp.graph.addEdge(5,3,4);
	Gp.from = 0;
	Gp.to = 5;

	DEBUG(Gp.from);
	DEBUG(Gp.to);
	AStar<GraphProblem> dk(Gp);
	assert(dk.foundGoal);
	for(int state = dk.last;;)
    {
        DEBUG(state);
        if(state == Gp.from) break;
        else state = *dk.pred.find(state);
    }
}

void testRBFS()
{
    typedef GraphAA<double> G;
    G sp;
    GraphProblem Gp;
	for(int i = 0; i < 6; ++i)
	{
		Gp.graph.addVertex();
	}
	Gp.graph.addEdge(0,1,6);
	Gp.graph.addEdge(0,2,8);
	Gp.graph.addEdge(0,3,18);
	Gp.graph.addEdge(1,4,11);
	Gp.graph.addEdge(2,3,9);
	Gp.graph.addEdge(4,5,3);
	Gp.graph.addEdge(5,2,7);
	Gp.graph.addEdge(5,3,4);
	Gp.from = 0;
	Gp.to = 5;

	RecursiveBestFirstSearch<GraphProblem> dk(Gp);
	assert(dk.foundGoal);
	while(!dk.pred.isEmpty())
    {
        DEBUG(dk.pred.pop());
    }
}

int main()
{
    testAStar();
    testRBFS();
    SmallTSPCompete();
    //testRBFSTSP();

    //TSPCompete();
    //testAStarTSP();*/
	return 0;
}
