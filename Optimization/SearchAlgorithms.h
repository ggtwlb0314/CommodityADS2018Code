#ifndef SEARCH_ALGORITHMS_H
#define SEARCH_ALGORITHMS_H
#include "../Utils/Vector.h"
#include "../HashTable/LinearProbingHashTable.h"
#include "../Heaps/Heap.h"
#include "../Sorting/Sort.h"
namespace igmdk{
//update!
template<typename PROBLEM> pair<bool, int> branchAndBound(PROBLEM& p,
    int maxLowerBounds = 1000000, bool foundCompleteSolution = false)
{//require one complete solution before stopping
    if(p.processSolution()) foundCompleteSolution = true;
    else if(!foundCompleteSolution || maxLowerBounds > 0)
    {
        Vector<pair<double, typename PROBLEM::Move> > moves =
            p.generateMoves();
        maxLowerBounds -= moves.getSize();
        quickSort(moves.getArray(), 0, moves.getSize() - 1,
            PairFirstComparator<double, typename PROBLEM::Move>());
        for(int i = 0; i < moves.getSize(); ++i)
            if(moves[i].first < p.getBestScore())
            {
                p.move(moves[i].second);
                pair<bool, int> status = branchAndBound(p, maxLowerBounds,
                    foundCompleteSolution);
                foundCompleteSolution = status.first;
                maxLowerBounds = status.second;
                p.undoMove(moves[i].second);
            }
            else break;//prune the rest
    }
    return make_pair(foundCompleteSolution, maxLowerBounds);
}

template<typename PROBLEM> struct AStar
{//closed set paths stored as parent pointer tree
    typedef typename PROBLEM::STATE_ID STATE_ID;
    typedef typename PROBLEM::HASHER HASHER;
    LinearProbingHashTable<STATE_ID, STATE_ID, HASHER> pred;
    bool foundGoal;
    STATE_ID last;//goal when found
    AStar(PROBLEM const& p, int maxSetSize = 1000000): foundGoal(false),
        last(p.start())
    {
        typedef pair<double, STATE_ID> QNode;
        IndexedHeap<QNode, PairFirstComparator<double, STATE_ID>, STATE_ID> pQ;
        STATE_ID j = last;
        //start has no predecessor
        pQ.insert(QNode(p.remainderLowerBound(j), PROBLEM::NULL_STATE()), j);
        while(!pQ.isEmpty() && pred.getSize() + pQ.getSize() < maxSetSize)
        {
            pair<QNode, STATE_ID> step = pQ.deleteMin();
            j = step.second;
            pred.insert(j, step.first.second);//now know best predecessor
            if(p.isGoal(j))
            {
                foundGoal = true;
                break;
            }//subtract the last move's lower bound to get the exact distance
            double dj = step.first.first - p.remainderLowerBound(j);
            Vector<STATE_ID> next = p.nextStates(j);
            for(int i = 0; i < next.getSize(); ++i)
            {
                STATE_ID to = next[i];
                double newChildLowerBound = dj + p.distance(j, to) +
                    p.remainderLowerBound(to);
                QNode const* current = pQ.find(to);
                if((current && newChildLowerBound < current->first) ||
                   (!current && !pred.find(to)))//update if better or new
                   pQ.changeKey(QNode(newChildLowerBound, j), to);
            }
        }
        last = j;
    }
};

template<typename PROBLEM> struct RecursiveBestFirstSearch
{
    typedef typename PROBLEM::STATE_ID STATE_ID;
    Stack<STATE_ID> pred;//path to the goal, which is top
    PROBLEM const& p;
    enum{SUCCESS = -1, FAILURE = -2};
    typedef pair<double, STATE_ID> INFO;//lower bound and state
    bool foundGoal;
    double work(INFO state, double alternative, double pathCost,
        int& maxLowerBounds)
    {//stop if found goal, of out of moves, or exceed computation budget
        if(p.isGoal(state.second)) return SUCCESS;
        Vector<STATE_ID> next = p.nextStates(state.second);
        if(next.getSize() == 0) return numeric_limits<double>::infinity();
        if(maxLowerBounds < next.getSize()) return FAILURE;
        maxLowerBounds -= next.getSize();
        //sort children by lower bound
        Heap<INFO, PairFirstComparator<double, STATE_ID> > children;
        for(int i = 0; i < next.getSize(); ++i)
            children.insert(INFO(max(state.first, pathCost +
                p.distance(state.second, next[i]) +
                p.remainderLowerBound(next[i])), next[i]));
        for(;;)
        {
            INFO best = children.deleteMin();
            //don't process remaining children if alternative better and
            //return the current best child value
            if(best.first > alternative) return best.first;
            pred.push(best.second);
            //as alternative use better of alternative and next best child
            best.first = work(best, children.isEmpty() ?
                alternative : min(children.getMin().first, alternative),
                pathCost + p.distance(state.second, best.second),
                maxLowerBounds);
            if(best.first == SUCCESS) return SUCCESS;
            else if(best.first == FAILURE) return FAILURE;
            children.insert(best);//enqueue child with revised estimate
            pred.pop();//undo move
        }
    }
    RecursiveBestFirstSearch(PROBLEM const& theProblem,
        int maxLowerBounds = 10000000): p(theProblem), foundGoal(false)
    {
        pred.push(p.start());
        foundGoal = (work(INFO(0.0, pred.getTop()),
            numeric_limits<double>::infinity(), 0, maxLowerBounds) == SUCCESS);
    }
};

}
#endif
