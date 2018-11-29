#ifndef METAHEURISTICS_H
#define METAHEURISTICS_H
#include "../Utils/Utils.h"
#include "../RandomNumberGeneration/Random.h"
namespace igmdk{

template<typename EVALUATOR> struct PermutationProblem
{
    Vector<int> current;
    EVALUATOR const& e;
    double currentScore;
    double evalStep(int from, int to)const//evaluate single step
        {return e.evalStep(current[from], current[to], to == 0);}
    PermutationProblem(Vector<int> const& initialOrder,
        EVALUATOR const& theE): e(theE)
    {
        assert(initialOrder.getSize() > 0);
        setCurrent(initialOrder);
    }
    Vector<int> const& getCurrent()const{return current;}
    void setCurrent(Vector<int> const& order)
    {
        current = order;
        currentScore = e(current);
    }
    double getCurrentScore()const{return currentScore;}
};
template<typename EVALUATOR> struct PermutationProblemReverseMove :
    public PermutationProblem<EVALUATOR>
{
    typedef PermutationProblem<EVALUATOR> PP;
    using PP::current;
    using PP::currentScore;
    using PP::evalStep;
    double evalReversal(int i, int j)const//incrementally evaluate
    {//take out edges i-1 to i and j-1 to j
     //add i-1 to j and i to j + 1
        int n = current.getSize();
        double im1Factor = i > 0 ? evalStep(i - 1, j) - evalStep(i - 1, i) : 0,
            jp1Factor = j + 1 < n ? evalStep(i, j + 1) -
            evalStep(j, j + 1) : 0;
        return -(im1Factor + jp1Factor);
    }
public:
    PermutationProblemReverseMove(Vector<int> const& initialOrder,
        EVALUATOR const& theE): PP(initialOrder, theE){}
    typedef Vector<int> MOVE;
    pair<MOVE, double> proposeMove()const//pick two random elements as
    {//reversal bounds - first must be not larger than second
        MOVE m = GlobalRNG().randomCombination(2, current.getSize());
        if(m[0] > m[1]) swap(m[0], m[1]);
        return make_pair(m, evalReversal(m[0], m[1]));
    }
    void applyMove(MOVE const& m)
    {
        assert(m.getSize() == 2);
        currentScore -= evalReversal(m[0], m[1]);
        current.reverse(m[0], m[1]);
    }
};
template<typename EVALUATOR> struct PermutationProblemSwapMove :
    public PermutationProblem<EVALUATOR>
{
    typedef PermutationProblem<EVALUATOR> PP;
    using PP::current;
    using PP::currentScore;
    using PP::evalStep;
    double evalSwap(int i, int j)const//incrementally evaluate
    {//take out edges i-1 to i and i to i + 1, j-1 to j and j to j + 1
     //add edges i-1 to j and j to i + 1, j-1 to i and i to j + 1
        int n = current.getSize();
        double im1Factor = i > 0 ? evalStep(i - 1, j) -
                evalStep(i - 1, i) : 0,
            ip1Factor = i + 1 < n ? evalStep(j, i + 1) -
                evalStep(i, i + 1) : 0,
            jm1Factor = j > 0? evalStep(j - 1, i) -
                evalStep(j - 1, j) : 0,
            jp1Factor = j + 1 < n ? evalStep(i, j + 1) -
                evalStep(j, j + 1) : 0;
        return -(im1Factor + ip1Factor + jm1Factor + jp1Factor);
    }
public:
    PermutationProblemSwapMove(Vector<int> const& initialOrder,
        EVALUATOR const& theE): PP(initialOrder, theE){}
    typedef Vector<int> MOVE;
    pair<MOVE, double> proposeMove()const//pick two random elements to swap
    {
        MOVE m = GlobalRNG().randomCombination(2, current.getSize());
        return make_pair(m, evalSwap(m[0], m[1]));
    }
    void applyMove(MOVE const& m)
    {
        assert(m.getSize() == 2);
        currentScore -= evalSwap(m[0], m[1]);
        swap(current[m[0]], current[m[1]]);
    }
};

template<typename PROBLEM> void localSearch(PROBLEM& p, int maxMoves,
    int maxStall)
{
    typedef pair<typename PROBLEM::MOVE, double> MOVE;
    for(int i = 0; maxMoves-- && i < maxStall; ++i)
    {
        MOVE m = p.proposeMove();
        if(m.second > 0)
        {
            i = -1;//reset counter on accept
            p.applyMove(m.first);
        }
    }
}

template<typename PROBLEM> void simulatedAnnealing(PROBLEM& p, double T,
    double coolingFactor, int maxMoves)
{
    typedef pair<typename PROBLEM::MOVE, double> MOVE;
    while(maxMoves--)
    {
        MOVE m = p.proposeMove();
        if(-m.second < T * GlobalRNG().exponential(1)) p.applyMove(m.first);
        T *= coolingFactor;
    }
}

template<typename PROBLEM>void iteratedLocalSearch(PROBLEM& p, int maxBigMoves)
{
    while(maxBigMoves--)
    {
        p.localSearchBest();
        p.updateBest();
        p.bigMove();
    }
}

}
#endif
