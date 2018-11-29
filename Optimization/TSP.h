#ifndef TSP_H
#define TSP_H
#include "../RandomNumberGeneration/Random.h"
#include "../ComputationalGeometry/Point.h"
#include "../Graphs/Graph.h"
#include "SearchAlgorithms.h"
#include "MetaHeuristics.h"
namespace igmdk{

struct TSPRandomInstance
{
    Vector<Vector<double> > points;
    TSPRandomInstance(int n)
    {
        for(int i = 0; i < n; ++i)
        {
            Vector<double> point;
            for(int j = 0; j < 2; ++j) point.append(GlobalRNG().uniform01());
            points.append(point);
        }
    }
    double evalStep(int from, int to, bool isReturn = false)const
    {//evaluate single step
        if(isReturn) return 0;//no cost to return
        assert(from >= 0 && from < points.getSize() && to >= 0 &&
            to < points.getSize());
        EuclideanDistance<Vector<double> >::Distance d;
        return d(points[from], points[to]);
    }
    double operator()(Vector<int> const& permutation)const
    {//evaluate complete solution
        assert(permutation.getSize() == points.getSize());
        double result = 0;
        for(int i = 1; i < points.getSize(); ++i)
            result += evalStep(permutation[i - 1], permutation[i]);
        return result;
    }
};

template<typename PROBLEM> double findMSTCost(PROBLEM const& instance,
    Vector<int> const& remPoints)
{
    int n = remPoints.getSize();
    if(n <= 1) return 0;
    GraphAA<double> g(n);
    for(int i = 0; i < n; ++i)
        for(int j = i + 1; j < n; ++j) g.addUndirectedEdge(i, j,
            instance.evalStep(remPoints[i], remPoints[j]));
    assert(validateGraph(g));
    Vector<int> parents = MST(g);
    double sum = 0;
    for(int i = 0; i < parents.getSize(); ++i)
    {
        int parent = parents[i];
        if(parent != -1)
            sum += instance.evalStep(remPoints[i], remPoints[parent]);
    }
    return sum;
}
template<typename PROBLEM> class BranchAndBoundPermutation
{
    Vector<int> permutation, best;
    PROBLEM const& problem;
    int n;
    double bestScore;
public:
    Vector<int> const& getBest()const{return best;}
    BranchAndBoundPermutation(int theN, PROBLEM const& theProblem): n(theN),
        problem(theProblem), bestScore(numeric_limits<double>::infinity()){}
    typedef int Move;
    bool processSolution()
    {
        if(permutation.getSize() < n) return false;
        double currentScore = problem(permutation);
        if(currentScore < bestScore)
        {
            best = permutation;
            bestScore = currentScore;
        }
        return true;
    }
    double getBestScore(){return bestScore;}
    Vector<pair<double, int> > generateMoves()
    {
        double sumNow = 0;
        Vector<bool> isIncluded(n, false);
        for(int i = 0; i < permutation.getSize(); ++i)
        {
            isIncluded[permutation[i]] = true;
            if(i > 0) sumNow += problem.evalStep(permutation[i - 1],
                permutation[i]);
        }
        Vector<pair<double, int> > result;
        for(int i = 0; i < n; ++i)
            if(!isIncluded[i])
            {
                Vector<int> remainder;
                for(int j = 0; j < n; ++j) if(j != i && !isIncluded[j])
                    remainder.append(j);
                double lb = sumNow + (permutation.getSize() > 0 ?
                    problem.evalStep(permutation.lastItem(), i) : 0) +
                    findMSTCost(problem, remainder);
                result.append(pair<double, int>(lb, i));
            }
        return result;
    }
    void move(int next){permutation.append(next);}
    void undoMove(int next){permutation.removeLast();}
};
template<typename INSTANCE> Vector<int> solveTSPBranchAndBound(INSTANCE const&
    instance, Vector<int> const& initialOrder, int maxLowerBounds)
{
    BranchAndBoundPermutation<INSTANCE> bb(instance.points.getSize(),
        instance);
    branchAndBound(bb, maxLowerBounds);
    return bb.getBest();
}

template<typename PROBLEM> struct AStartTSPProblem
{//Enhancement - allow path compression using pred in AStar !?
    PROBLEM const& problem;
    typedef Vector<int> STATE_ID;//state is current permutation
    static STATE_ID NULL_STATE(){return STATE_ID();}
    typedef DataHash<> HASHER;
    //don't know best first node for no-return case
    STATE_ID start()const{return NULL_STATE();}
    Vector<int> findRemainder(STATE_ID const& id)const
    {
        int n = problem.points.getSize();
        Vector<bool> included(n, false);
        for(int i = 0; i < id.getSize(); ++i)  included[id[i]] = true;
        Vector<int> result;
        for(int i = 0; i < n; ++i) if(!included[i]) result.append(i);
        return result;
    }
    bool isGoal(STATE_ID const& i)const
    {
        Vector<int> remainder = findRemainder(i);
        return remainder.getSize() == 0;
    }
    Vector<STATE_ID> nextStates(STATE_ID const& j)const
    {
        Vector<STATE_ID> result;
        Vector<int> remainder = findRemainder(j);
        for(int i = 0; i < remainder.getSize(); ++i)
        {
            STATE_ID to = j;
            to.append(remainder[i]);
            result.append(to);
        }
        return result;
    }
    double remainderLowerBound(STATE_ID const& id)const
        {return findMSTCost(problem, findRemainder(id));}
    double distance(STATE_ID const& j, STATE_ID const& to)const
    {//no cost to start + assume longer state has all information
        STATE_ID const& longer = j.getSize() < to.getSize() ? to : j;
        return longer.getSize() > 1 ?
            problem.evalStep(j.lastItem(), to.lastItem()) : 0;
    }
};
template<typename INSTANCE> pair<Vector<int>, bool> solveTSPAStar(INSTANCE
    const& instance, int maxSetSize)
{
    AStartTSPProblem<INSTANCE> p = {instance};
    AStar<AStartTSPProblem<INSTANCE> > as(p, maxSetSize);
    if(as.foundGoal)//success
    {
        Vector<int> path = as.last;
        path.reverse();//need path in travel not parent pointer order
        return make_pair(path, true);
    }
    return make_pair(Vector<int>(), false);//failure
}
template<typename INSTANCE> pair<Vector<int>, bool> solveTSPRBFS(INSTANCE
    const& instance, int maxLowerBounds)
{
    AStartTSPProblem<INSTANCE> p = {instance};
    RecursiveBestFirstSearch<AStartTSPProblem<INSTANCE> > rbfs(p,
        maxLowerBounds);
    if(rbfs.foundGoal)//success
    {
        Vector<int> path = rbfs.pred.getTop();
        path.reverse();//need path in travel not parent pointer order
        return make_pair(path, true);
    }
    return make_pair(Vector<int>(), false);//failure
}

template<typename MOVE, typename INSTANCE> Vector<int> solveTSPLocalSearch(
    INSTANCE const& instance, Vector<int> const& initialOrder, int maxMoves,
    int maxStall)
{
    MOVE m(initialOrder, instance);
    localSearch(m, maxMoves, maxStall);
    return m.getCurrent();
}

template<typename MOVE, typename INSTANCE> Vector<int>
    solveTSPSimulatedAnnealing(INSTANCE const& instance, Vector<int> const&
    initialOrder, double T, double coolingFactor, int maxMoves)
{
    MOVE m(initialOrder, instance);
    simulatedAnnealing(m, T, coolingFactor, maxMoves);
    return m.getCurrent();
}

template<typename EVALUATOR> struct TSPILSFromRandReverseMove
{
    PermutationProblemReverseMove<EVALUATOR> rm;
    Vector<int> best;
    EVALUATOR const& e;
    double bestScore;
    int lsMoves, lsStall;
    TSPILSFromRandReverseMove(Vector<int> const& initialOrder,
        EVALUATOR const& theE, int lsMaxMoves, int lsMaxStall):
        lsMoves(lsMaxMoves), lsStall(lsMaxStall), rm(initialOrder, theE),
        best(initialOrder), e(theE), bestScore(theE(initialOrder))
        {assert(initialOrder.getSize() > 0);}
    void localSearchBest(){localSearch(rm, twoPower(10), 100);}
    void updateBest()
    {
        if(rm.getCurrentScore() < bestScore)
        {
            bestScore = rm.getCurrentScore();
            best = rm.getCurrent();
        }
    }
    void bigMove()
    {
        updateBest();
        Vector<int> next = best;
        GlobalRNG().randomPermutation(next.getArray(), next.getSize());
        rm.setCurrent(next);
    }
};
template<typename INSTANCE> Vector<int> solveTSPIteratedLocalSearch(INSTANCE
    const& instance, Vector<int> const& initialOrder, int lsMaxMoves,
    int lsMaxStall, int bigMoves)
{
    TSPILSFromRandReverseMove<TSPRandomInstance> move(initialOrder,
        instance, lsMaxMoves, lsMaxStall);
    iteratedLocalSearch(move, bigMoves);
    return move.best;
}

}
#endif
