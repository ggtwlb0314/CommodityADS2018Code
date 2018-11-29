#ifndef GRAPH_H
#define GRAPH_H
#include "../Utils/Vector.h"
#include "../Utils/Debug.h"
#include "../RandomNumberGeneration/Random.h"
#include "../Utils/Queue.h"
#include "../Utils/UnionFind.h"
#include <cassert>
#include "../Heaps/Heap.h"
using namespace std;
namespace igmdk{

template<typename EDGE_DATA> class GraphAA
{
    struct Edge
    {
        int to;
        EDGE_DATA edgeData;
        Edge(int theTo, EDGE_DATA const& theEdgeData): to(theTo),
            edgeData(theEdgeData) {}
    };
    Vector<Vector<Edge> > vertices;
public:
    GraphAA(int initialSize = 0): vertices(initialSize) {}
    int nVertices()const{return vertices.getSize();}
    int nEdges(int v)const{return vertices[v].getSize();}
    void addVertex(){vertices.append(Vector<Edge>());}
    void addEdge(int from, int to, EDGE_DATA const& edgeData = EDGE_DATA())
    {
        assert(to >= 0 && to < vertices.getSize());
        vertices[from].append(Edge(to, edgeData));
    }
    void addUndirectedEdge(int from, int to,
        EDGE_DATA const& edgeData = EDGE_DATA())
    {
        addEdge(from, to, edgeData);
        addEdge(to, from, edgeData);
    }
    class AdjacencyIterator
    {
        Vector<Edge> const* edges;
        int j;//current edge
    public:
        AdjacencyIterator(GraphAA const& g, int v, int theJ):
            edges(&g.vertices[v]), j(theJ){}
        AdjacencyIterator& operator++()
        {
            assert(j < edges->getSize());
            ++j;
            return *this;
        }
        int to(){return (*edges)[j].to;}
        EDGE_DATA const& data(){return (*edges)[j].edgeData;}
        bool operator!=(AdjacencyIterator const& rhs){return j != rhs.j;}
    };
    AdjacencyIterator begin(int v)const
        {return AdjacencyIterator(*this, v, 0);}
    AdjacencyIterator end(int v)const
        {return AdjacencyIterator(*this, v, nEdges(v));}
};

template<typename GRAPH> bool validateGraph(GRAPH const& g)
{//check for repeated and self edges
    for(int i = 0; i < g.nVertices(); ++i)
    {
        Vector<bool> marked(g.nVertices(), false);
        for(typename GRAPH::AdjacencyIterator j = g.begin(i); j != g.end(i);
            ++j)
            if(j.to() == i || marked[j.to()]) return false;
            else marked[j.to()] = true;
    }
    return true;
}

template<typename GRAPH> GRAPH reverse(GRAPH const& g)
{
    GRAPH result(g.nVertices());
    for(int i = 0; i < g.nVertices(); ++i)
        for(typename GRAPH::AdjacencyIterator j = g.begin(i);
            j != g.end(i); ++j) result.addEdge(j.to(), i, j.data());
    return result;
}

template<typename GRAPH>
GRAPH randomDirectedGraph(int vertices, int edgesPerVertex)
{
    assert(edgesPerVertex <= vertices);
    GRAPH g(vertices);
    for(int i = 0; i < vertices; ++i)
    {
        Vector<int> edges = GlobalRNG().sortedSample(edgesPerVertex, vertices);
        for(int j = 0; j < edgesPerVertex; ++j) g.addEdge(i, edges[i]);
    }
    return g;
}

struct DefaultDFSAction
{
    void source(int v){}
    void treeEdge(int v){}
    void nonTreeEdge(int v){}
    void backwardEdge(int v){}
};
template<typename GRAPH, typename ACTION> void DFSComponent(GRAPH const& g,
    int source, Vector<bool>& visited, ACTION& a = ACTION())
{
    typedef typename GRAPH::AdjacencyIterator ITER;
    Stack<pair<ITER, int> > s;
    s.push(make_pair(g.begin(source), source));
    while(!s.isEmpty())
    {
        ITER& j = s.getTop().first;
        if(j != g.end(s.getTop().second))
        {
            int to = j.to();
            if(visited[to]) a.nonTreeEdge(to);
            else
            {
                a.treeEdge(to);
                visited[to] = true;
                s.push(make_pair(g.begin(to), to));
            }
            ++j;
        }
        else
        {
            s.pop();
            if(!s.isEmpty()) a.backwardEdge(s.getTop().second);
        }
    }
}

template<typename GRAPH, typename ACTION> void DFS(GRAPH const& g,
    ACTION& a = ACTION())
{
    Vector<bool> visited(g.nVertices(), false);
    for(int i = 0; i < g.nVertices(); ++i) if(!visited[i])
    {
        a.source(i);
        visited[i] = true;
        DFSComponent(g, i, visited, a);
    }
}

struct ConnectedComponentAction: public DefaultDFSAction
{
    Vector<Vector<int> > components;
    void source(int v)
    {
        components.append(Vector<int>());
        treeEdge(v);
    }
    void treeEdge(int v){components.lastItem().append(v);}
};
template<typename GRAPH>
Vector<Vector<int> > connectedComponents(GRAPH const& g)
{
    ConnectedComponentAction a;
    DFS(g, a);
    return a.components;
}

struct TopologicalSortAction
{
    int currentRank, leaf;//current DFS tree leaf
    Vector<int> ranks;
    bool hasCycle;
    TopologicalSortAction(int nVertices): currentRank(nVertices), leaf(-1),
        ranks(nVertices, -1), hasCycle(false) {}
    void source(int v){treeEdge(v);}
    void treeEdge(int v){leaf = v;}//potential leaf
    //unranked v = cross edge
    void nonTreeEdge(int v){if(ranks[v] == -1) hasCycle = true;}
    void backwardEdge(int v)
    {
        if(leaf != -1)
        {//assign rank to DFS tree leaf if any
            ranks[leaf] = --currentRank;
            leaf = -1;
        }
        ranks[v] = --currentRank;
    }
};
template<typename GRAPH> Vector<int> topologicalSort(GRAPH const& g)
{
    TopologicalSortAction a(g.nVertices());
    DFS(g, a);
    if(a.hasCycle) a.ranks = Vector<int>();//empty ranks signals cycle
    return a.ranks;
}

template<typename GRAPH> Vector<int> BFS(GRAPH& g, int source)
{
    Vector<int> distances(g.nVertices(), -1);
    Queue<int> q(g.nVertices());
    distances[source] = 0;
    q.push(source);
    while(!q.isEmpty())
    {
        int i = q.pop();
        for(typename GRAPH::AdjacencyIterator j = g.begin(i); j != g.end(i);
            ++j) if(distances[j.to()] == -1)
            {
                distances[j.to()] = distances[i] + 1;
                q.push(j.to());
            }
    }
    return distances;
}

struct StaticGraph
{
    //does not need edge and vertex data, easy to iterate
    //inherently directed, for indirection needs coordination
    //by caller
    Vector<int> edges, starts;
    StaticGraph(){starts.append(0);}
    void addVertex()
    {
        starts.append(starts.lastItem());
        starts[starts.getSize()-2] = 0;
    }
    void addEdgeToLastVertex(int to)
    {
        edges.append(to);
        ++starts.lastItem();
    }
    int getSize(){return starts.getSize()-1;}
};

template<typename GRAPH> Vector<int> MST(GRAPH& g)
{//represent MST as edges to parent vertices (first node won't have any)
    Vector<int> parents(g.nVertices(), -1);
    typedef pair<double, int> QNode;
    IndexedArrayHeap<QNode, PairFirstComparator<double, int> > pQ;
    for(int i = 0; i < g.nVertices(); ++i)
        pQ.insert(QNode(numeric_limits<double>::max(), i), i);
    while(!pQ.isEmpty())
    {
        int i = pQ.deleteMin().first.second;
        for(typename GRAPH::AdjacencyIterator j = g.begin(i); j != g.end(i);
            ++j)
        {//adjust best known distances to child vertices not yet in the tree
            int to = j.to();
            QNode const* child = pQ.find(to);//child may no longer be in q
            if(child && j.data() < child->first)
            {
                pQ.changeKey(QNode(j.data(), to), to);
                parents[j.to()] = i;//update to closer parent
            }
        }
    }
    return parents;
}

template<typename GRAPH>
Vector<int> ShortestPath(GRAPH& g, int from, int dest = -1)
{
    assert(from >= 0 && from < g.nVertices());
    Vector<int> pred(g.nVertices(), -1);
    typedef pair<double, int> QNode;
    IndexedArrayHeap<QNode, PairFirstComparator<double, int> > pQ;
    for(int i = 0; i < g.nVertices(); ++i) pQ.insert(
        QNode(i == from ? 0 : numeric_limits<double>::infinity(), i), i);
    while(!pQ.isEmpty() && pQ.getMin().first.second != dest)
    {
        int i = pQ.getMin().first.second;
        double dj = pQ.deleteMin().first.first;//distance to the current node
        for(typename GRAPH::AdjacencyIterator j = g.begin(i); j != g.end(i);
            ++j)
        {//child may no longer be in q
            double newChildDistance = dj + j.data();
            int to = j.to();
            QNode const* child = pQ.find(to);
            if(child && newChildDistance < child->first)
            {
                pQ.changeKey(QNode(newChildDistance, to), to);
                pred[to] = i;//new best parent
            }
        }
    }
    return pred;
}

template<typename GRAPH> struct BellmanFord
{
    int v;//must be first
    Vector<double> distances;
    Vector<int> pred;
    bool hasNegativeCycle;
    bool findNegativeCycle()
    {
        UnionFind uf(v);
        for(int i = 0; i < v; ++i)
        {
            int parent = pred[i];
            if(parent != -1)
            {//can't be in same subset as parent before join
                if(uf.areEquivalent(i, parent)) return true;
                uf.join(i, parent);
            }
        }
        return false;
    }
    BellmanFord(GRAPH& g, int from): v(g.nVertices()), pred(v, -1),
        distances(v, numeric_limits<double>::infinity()),
        hasNegativeCycle(false)
    {
        assert(from >= 0 && from < v);
        Queue<int> queue;
        Vector<bool> onQ(v, false);
        distances[from] = 0;
        queue.push(from);
        onQ[from] = true;
        for(int nIterations = 0; !queue.isEmpty() && !hasNegativeCycle;)
        {
            int i = queue.pop();
            onQ[i] = false;
            for(typename GRAPH::AdjacencyIterator j = g.begin(i);
                j != g.end(i); ++j)
            {
                double newChildDistance = distances[i] + j.data();
                if(newChildDistance < distances[j.to()])
                {
                    distances[j.to()] = newChildDistance;
                    pred[j.to()] = i;//new best parent
                    if(!onQ[j.to()])
                    {
                        queue.push(j.to());
                        onQ[j.to()] = true;
                    }
                }//check for negative cycles every v inner iterations
                if(++nIterations % v == 0)
                    hasNegativeCycle = findNegativeCycle();
            }
        }
    }
};

struct FlowData
{
    int from;
    double flow, capacity, cost;//cost used only for min flow
    FlowData(int theFrom, double theCapacity, double theCost = 0):
        from(theFrom), capacity(theCapacity), flow(0), cost(theCost) {}
    double capacityTo(int v)const{return v == from ? flow : capacity - flow;}
    //flow can step out of (0, capacity) numerically but OK
    void addFlowTo(int v, double change)
        {flow += change * (v == from ? -1 : 1);}
};
template<typename GRAPH> class ShortestAugmentingPath
{
    int v;
    Vector<int> path, pred;
    double totalFlow;
    bool hasAugmentingPath(GRAPH const& g, Vector<FlowData>& fedges, int from,
        int to)
    {
        for(int i = 0; i < v; ++i) pred[i] = -1;
        Queue<int> queue;
        queue.push(pred[from] = from);
        while(!queue.isEmpty())
        {
            int i = queue.pop();
            for(typename GRAPH::AdjacencyIterator j = g.begin(i);
                j != g.end(i); ++j)
                if(pred[j.to()] == -1 &&//unvisited with capacity
                    fedges[j.data()].capacityTo(j.to()) > 0)
                {
                    pred[j.to()] = i;
                    path[j.to()] = j.data();
                    queue.push(j.to());
                }
        }
        return pred[to] != -1;
    }
    bool hasMinCostAugmentingPath(GRAPH const& g, Vector<FlowData>& fedges,
        int from, int to, double neededFlow)
    {//if need more flow, make a graph from edges with available capacity
        if(totalFlow >= abs(neededFlow)) return false;
        GraphAA<double> costGraph(v);
        for(int i = 0; i < v; ++i)
            for(typename GRAPH::AdjacencyIterator j = g.begin(i);
                j != g.end(i); ++j)
                if(fedges[j.data()].capacityTo(j.to()) > 0)
                    costGraph.addEdge(i, j.to(), fedges[j.data()].cost);
        if(neededFlow > 0) pred = ShortestPath(costGraph, from, to);
        else
        {//negative costs
            BellmanFord<GraphAA<double> > bf(costGraph, from);
            if(bf.hasNegativeCycle)
            {
                totalFlow = numeric_limits<double>::infinity();
                return false;
            }
            pred = bf.pred;
        };
        //extract edges for the path
        for(int i = to; pred[i] != -1; i = pred[i])
            for(typename GRAPH::AdjacencyIterator j = g.begin(pred[i]);
                j != g.end(pred[i]); ++j)
                if(j.to() == i)
                {
                    path[i] = j.data();
                    break;
                }
        return pred[to] != -1;
    }
public:
    double getTotalFlow()const{return totalFlow;}
    ShortestAugmentingPath(GRAPH const& g, Vector<FlowData>& fedges, int from,
        int to, double neededFlow = 0): v(g.nVertices()), totalFlow(0),
        path(v, -1), pred(v, -1)
    {//iteratively, first find a path
        assert(from >= 0 && from < v && to >= 0 && to < v);
        while(neededFlow == 0 ? hasAugmentingPath(g, fedges, from, to) :
            hasMinCostAugmentingPath(g, fedges, from, to, neededFlow))
        {//then from it the amount of flow to add
            double increment = numeric_limits<double>::max();
            for(int j = to; j != from; j = pred[j])
                increment = min(increment, fedges[path[j]].capacityTo(j));
            if(neededFlow != 0)//only relevant to min cost flow
                increment = min(increment, abs(neededFlow) - totalFlow);
            //then add to all edges
            for(int j = to; j != from; j = pred[j])
                fedges[path[j]].addFlowTo(j, increment);
            totalFlow += increment;
        }
    }
};

Vector<pair<int, int> > bipartiteMatching(int n, int m,
    Vector<pair<int, int> > const& allowedMatches)
{//v = n + m + 2, e = n + m + allowedMatches.getSize(), time is O(ve)
    GraphAA<int> sp(n + m + 2);//setup graph and flow edges
    Vector<FlowData> data;
    for(int i = 0; i < allowedMatches.getSize(); ++i)
    {
        data.append(FlowData(allowedMatches[i].first, 1));
        sp.addUndirectedEdge(allowedMatches[i].first,
            allowedMatches[i].second, i);
    }
    int source = n + m, sink = source + 1;//setup source and sink groups
    for(int i = 0; i < source; ++i)
    {
        int from = i, to = sink;
        if(i < n)
        {
            from = source;
            to = i;
        }
        data.append(FlowData(from, 1));
        sp.addUndirectedEdge(from, to, i + allowedMatches.getSize());
    }//calculate the matching
    ShortestAugmentingPath<GraphAA<int> > dk(sp, data, source, sink);
    //return edges with positive flow
    Vector<pair<int, int> > result;
    for(int i = 0; i < allowedMatches.getSize(); ++i)
        if(data[i].flow > 0) result.append(allowedMatches[i]);
    return result;
}

Vector<int> stableMatching(Vector<Vector<int> > const& womenOrders,
    Vector<Vector<int> > const& menScores)
{
    int n = womenOrders.getSize(), m = menScores.getSize();
    assert(n <= m);
    Stack<int> unassignedMen;//any list type will do
    for(int i = 0; i < n; ++i) unassignedMen.push(i);
    Vector<int> currentMan(m, -1), nextWoman(n, 0);
    while(!unassignedMen.isEmpty())
    {
        int man = unassignedMen.pop(), woman, currentM;
        do
        {//won't run out of bounds due to n <= m
            woman = nextWoman[man]++;
            currentM = currentMan[woman];
        }while(currentM != -1 &&//man finds best woman that prefers him
            menScores[woman][man] <= menScores[woman][currentM]);
        currentMan[woman] = man;//found match
        //divorcee, if any, to search more
        if(currentM != -1) unassignedMen.push(currentM);
    }
    return currentMan;
}

Vector<pair<int, int> > assignmentProblem(int n, int m,
    Vector<pair<pair<int, int>, double> > const& allowedMatches)
{//v = n + m + 2, e = n + m + allowedMatches.getSize()
    GraphAA<int> sp(n + m + 2);//setup graph and flow edges
    Vector<FlowData> data;
    for(int i = 0; i < allowedMatches.getSize(); ++i)
    {
        data.append(FlowData(allowedMatches[i].first.first, 1,
            allowedMatches[i].second));
        sp.addUndirectedEdge(allowedMatches[i].first.first,
            allowedMatches[i].first.second, i);
    }
    int source = n + m, sink = source + 1;//setup source and sink groups
    for(int i = 0; i < source; ++i)
    {
        int from = i, to = sink;
        if(i < n)
        {
            from = source;
            to = i;
        }
        data.append(FlowData(from, 1, 0));
        sp.addUndirectedEdge(from, to, i + allowedMatches.getSize());
    }//calculate the matching
    ShortestAugmentingPath<GraphAA<int> > dummy(sp, data, source, sink,
        min(n, m));
    //return edges with positive flow
    Vector<pair<int, int> > result;
    for(int i = 0; i < allowedMatches.getSize(); ++i)
        if(data[i].flow > 0) result.append(allowedMatches[i].first);
    return result;
}

}
#endif
