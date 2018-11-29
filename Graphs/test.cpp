#include "Graph.h"
#include "../Utils/Debug.h"
using namespace igmdk;

void testMST()
{
    typedef GraphAA<double> G;
    G sp;
    for(int i = 0; i < 5; ++i)
	{
		sp.addVertex();
	}
	sp.addEdge(0,1,2);
	sp.addEdge(0,3,8);
	sp.addEdge(0,4,4);
	sp.addEdge(1,0,2);
	sp.addEdge(1,2,3);
	sp.addEdge(2,1,3);
	sp.addEdge(2,3,5);
	sp.addEdge(2,4,1);
	sp.addEdge(3,0,8);
	sp.addEdge(3,2,5);
	sp.addEdge(3,4,7);
	sp.addEdge(4,0,4);
	sp.addEdge(4,2,1);
	sp.addEdge(4,3,7);

    Vector<int> parents = MST(sp);
    for(int i = 0; i < parents.getSize(); ++i)
    {
        DEBUG(parents[i]);
    }
}

void testShortestPath()
{
    typedef GraphAA<double> G;
    G sp;
	for(int i = 0; i < 6; ++i)
	{
		sp.addVertex();
	}
	sp.addEdge(0,1,6);
	sp.addEdge(0,2,8);
	sp.addEdge(0,3,18);
	sp.addEdge(1,4,11);
	sp.addEdge(2,3,9);
	sp.addEdge(4,5,3);
	sp.addEdge(5,2,7);
	sp.addEdge(5,3,4);

	Vector<int> pred = ShortestPath(sp, 0);
	//BellmanFord<G> dk(sp, 0);
	for(int i = 0; i < pred.getSize(); ++i)
    {
        DEBUG(pred[i]);
    }
}

void testGraph()
{
    typedef GraphAA<bool> G;
    G sp;
    for(int i = 0; i < 6; ++i)
	{
		sp.addVertex();
	}
	sp.addEdge(0,1,6);
	sp.addEdge(0,2,8);
	sp.addEdge(0,3,18);
	sp.addEdge(1,4,11);
	sp.addEdge(2,3,9);
	sp.addEdge(4,5,3);
	sp.addEdge(5,2,7);
	sp.addEdge(5,3,4);

    Vector<Vector<int> > components = connectedComponents(sp);
    DEBUG(components.getSize());
    for(int i = 0; i < components.getSize(); ++i)
    {
        Vector<int>& c = components[i];
        for(int j = 0; j < c.getSize(); ++j)
        {
            DEBUG(c[j]);
        }
    }

    Vector<int> ranks = topologicalSort(sp);
    DEBUG(ranks.getSize());
    for(int j = 0; j < ranks.getSize(); ++j)
    {
        DEBUG(ranks[j]);
    }
}

void testGraph2()
{
    typedef GraphAA<bool> G;
    G sp;
    for(int i = 0; i < 6; ++i)
	{
		sp.addVertex();
	}
	sp.addEdge(0,1,6);
	sp.addEdge(0,2,8);
	sp.addEdge(0,3,18);
	sp.addEdge(1,4,11);
	sp.addEdge(2,3,9);
	sp.addEdge(4,5,3);
	sp.addEdge(5,2,7);
	sp.addEdge(5,3,4);

    Vector<int> distances = BFS(sp, 0);
    for(int j = 0; j < distances.getSize(); ++j)
    {
        DEBUG(distances[j]);
    }
}

void timeSRT()
{
    typedef GraphAA<double> G;
    G sp;
	for(int i = 0; i < 6; ++i)
	{
		sp.addVertex();
	}
	sp.addEdge(0,1,6);
	sp.addEdge(0,2,8);
	sp.addEdge(0,3,18);
	sp.addEdge(1,4,11);
	sp.addEdge(2,3,9);
	sp.addEdge(4,5,3);
	sp.addEdge(5,2,7);
	sp.addEdge(5,3,4);

	BellmanFord<G> dk(sp, 0);
	for(int i = 0; i < dk.pred.getSize(); ++i)
    {
        DEBUG(dk.pred[i]);
    }
}

void timeFF()
{
    GraphAA<int> sp(6);
    Vector<FlowData> data;
    data.append(FlowData(0, 2));//to 1
	sp.addUndirectedEdge(0,1,0);
	data.append(FlowData(0, 3));//to 2
	sp.addUndirectedEdge(0,2,1);
	data.append(FlowData(1, 3));//to 3
	sp.addUndirectedEdge(1,3,2);
	data.append(FlowData(1, 1));//to 4
	sp.addUndirectedEdge(1,4,3);
	data.append(FlowData(2, 1));//to 3
	sp.addUndirectedEdge(2,3,4);
	data.append(FlowData(2, 1));//to 4
	sp.addUndirectedEdge(2,4,5);
	data.append(FlowData(3, 2));//to 5
	sp.addUndirectedEdge(3,5,6);
	data.append(FlowData(4, 3));//to 5
	sp.addUndirectedEdge(4,5,7);

	ShortestAugmentingPath<GraphAA<int> > dk(sp, data, 0, 5);
	for(int i = 0; i < data.getSize(); ++i)
    {
        DEBUG(data[i].flow);
        DEBUG(data[i].capacity);
    }
    DEBUG(dk.getTotalFlow());
}



void timeFFMinCost()
{
    GraphAA<int> sp(6);
    Vector<FlowData> data;
    data.append(FlowData(0, 300, 0));//to 1
	sp.addUndirectedEdge(0,1,0);
	data.append(FlowData(0, 300, 0));//to 2
	sp.addUndirectedEdge(0,2,1);
	data.append(FlowData(1, 200, 7));//to 3
	sp.addUndirectedEdge(1,3,2);
	data.append(FlowData(1, 200, 6));//to 4
	sp.addUndirectedEdge(1,4,3);
	data.append(FlowData(2, 280, 4));//to 3
	sp.addUndirectedEdge(2,3,4);
	data.append(FlowData(2, 350, 6));//to 4
	sp.addUndirectedEdge(2,4,5);
	data.append(FlowData(3, 300, 0));//to 5
	sp.addUndirectedEdge(3,5,6);
	data.append(FlowData(4, 300, 0));//to 5
	sp.addUndirectedEdge(4,5,7);

	ShortestAugmentingPath<GraphAA<int> > dk(sp, data, 0, 5, 600);
	for(int i = 0; i < data.getSize(); ++i)
    {
        DEBUG(data[i].flow);
        DEBUG(data[i].capacity);
    }
    DEBUG(dk.getTotalFlow());
}

void testBM()
{
    Vector<pair<int, int> > allowed;
    allowed.append(make_pair(0, 5));
    allowed.append(make_pair(1, 4));
    allowed.append(make_pair(1, 3));
    allowed.append(make_pair(2, 4));
    allowed = bipartiteMatching(3, 3, allowed);
	for(int i = 0; i < allowed.getSize(); ++i)
    {
        DEBUG(allowed[i].first);
        DEBUG(allowed[i].second);
    }
}

void testAssignment()
{
    Vector<pair<pair<int, int>, double> > allowed;
    allowed.append(make_pair(make_pair(0, 5), 0));
    allowed.append(make_pair(make_pair(1, 4), 0));
    allowed.append(make_pair(make_pair(1, 3), 0));
    allowed.append(make_pair(make_pair(2, 4), 0));
    Vector<pair<int, int> > allowed2 = assignmentProblem(3, 3, allowed);
	for(int i = 0; i < allowed2.getSize(); ++i)
    {
        DEBUG(allowed2[i].first);
        DEBUG(allowed2[i].second);
    }
}

void testStableMatching()
{
    Vector<Vector<int> > womenOrder, menRanks;
    Vector<int> order1, order2, rank1, rank2;
    order1.append(0);
    order1.append(1);
    order2.append(1);
    order2.append(0);
    womenOrder.append(order1);
    womenOrder.append(order2);
    rank1.append(0);
    rank1.append(1);
    rank2.append(1);
    rank2.append(0);
    menRanks.append(rank1);
    menRanks.append(rank2);
    Vector<int> womenResult = stableMatching(womenOrder, menRanks);
    for(int i = 0; i < womenResult.getSize(); ++i)
    {
        DEBUG(womenResult[i]);
    }
}

void DDDGraph()
{
    typedef GraphAA<double> G;
    G Graph05;
	for(int i = 0; i < 6; ++i)
	{
		Graph05.addVertex();
	}
	Graph05.addEdge(0,1,6);
	Graph05.addEdge(0,2,8);
	Graph05.addEdge(0,3,18);
	Graph05.addEdge(1,4,11);
	Graph05.addEdge(2,3,9);
	Graph05.addEdge(4,5,3);
	Graph05.addEdge(5,2,7);
	Graph05.addEdge(5,3,4);

	cout << "breakpoint" << endl;
}

int main()
{
    timeSRT();
    return 0;
    DDDGraph();

    testMST();
    testShortestPath();
    testGraph();
    testGraph2();
	clock_t start = clock();
	int N = 150000;
	timeFF();
	testAssignment();
	testBM();
	int tFL = (clock() - start);
    timeFFMinCost();
	testStableMatching();
    cout << "FL: "<<tFL << endl;
	return 0;
}
