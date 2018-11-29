#include <iostream>
#include <cmath>
#include "KDTree.h"
#include "Point.h"
#include "ComputationalGeometryTestAuto.h"
#include "../RandomNumberGeneration/Random.h"
#include "../NumericalMethods/NumericalMethods.h"
#include "../HashTable/ChainingHashTable.h"
#include "../RandomNumberGeneration/Random.h"
#include "../RandomNumberGeneration/Statistics.h"
using namespace igmdk;

template<typename KEY, typename VALUE, typename DISTANCE> class KNNBruteForce
{
    DISTANCE distance;
    typedef KVPair<KEY, VALUE> Node;
    Vector<Node> nodes;
    struct QNode
    {
        double distance;
        int result;
        bool operator<(QNode const& rhs)const
            {return distance > rhs.distance;}
    };
public:
    KNNBruteForce(DISTANCE const& theDistance = DISTANCE()):
        distance(theDistance){}
    typedef Node NodeType;
    void insert(KEY const& key, VALUE const& value)
        {nodes.append(Node(key, value));}
    Vector<NodeType*> kNN(KEY const& key, int k)
    {
        Heap<QNode> q;
        for(int i = 0; i < nodes.getSize(); ++i)
        {
            QNode node = {distance(key, nodes[i].key), i};
            if(q.getSize() < k) q.insert(node);
            else if(node.distance < q.getMin().distance)
                q.changeKey(0, node);
        }
        Vector<NodeType*> result;
        while(!q.isEmpty()) result.append(&nodes[q.deleteMin().result]);
        result.reverse();
        return result;
    }
    NodeType* nearestNeighbor(KEY const& key){return kNN(key, 1)[0];}
};

class E2LSHHasher
{
    Vector<EHash<PrimeHash> > mappers;
    struct Hasher
    {
        Vector<double> a;
        double w, b;
        Hasher(int D, int r): w(r), b(GlobalRNG().uniform01() * w)
            {for(int i = 0; i < D; ++i) a.append(GlobalRNG().normal01());}
        int operator()(Vector<double> const& x)const
            {return int((dotProduct(a, x) + b)/w);}
    };
    Vector<Hasher> h;
public:
    typedef unsigned long long RESULT_TYPE;
    typedef Vector<double> ITEM_TYPE;
    E2LSHHasher(int k, int l, int D, double w): mappers(l)
        {for(int i = 0; i < k * l; ++i) h.append(Hasher(D, w));}
    RESULT_TYPE operator()(ITEM_TYPE const& x, int bucket)const
    {
        Vector<int> result;
        int k = h.getSize()/mappers.getSize();
        for(int i = 0; i < k; ++i) {result.append(h[bucket * k + i](x)); //DEBUG(result.lastItem());
        }
        //system("PAUSE");
        return mappers[bucket](result.getArray(), result.getSize());
    }
    static double p(double w, double r)
    {
        double z = r/w;
        return 2 * approxNormalCDF(z) - 1 -
            2/sqrt(2 * PI())/z * (1 - exp(-z * z/2));
    }
    static double p1(double r){return p(r, r);}
    static double p2(double r, double c){return p(r, r * c);}
    static double distance(ITEM_TYPE const& x1, ITEM_TYPE const& x2)
    {
        EuclideanDistance<ITEM_TYPE>::Distance ed;
        return ed(x1, x2);
    }
};

namespace LSHKLFinder
{
    int LSHGetL(int k, double p1, double e)
    {
        double l = log(e)/log(1 - pow(p1, k));
        return (!isfinite(l) && l >= numeric_limits<int>::max()) ? -1 : 1 + int(l);
    }
    double LSHCost(int k, double e, double p1, double p2, int n)
    {
        int l = LSHGetL(k, p1, e);
        return (10 * k + pow(p2, k) * n) * l;
    }
    int minL(double p1, double e){return LSHGetL(1, p1, e);};
    int LSHFindK(double e, double p1, double p2, int n, int maxL)
    {
        int bestK = -1;
        double bestV;
        for(int k = 1;; ++k)
        {
            DEBUG(k);
            int l = LSHGetL(k, p1, e);
            DEBUG(l);
            double v = LSHCost(k, e, p1, p2, n);
            DEBUG(v);
            if(v < 0) break;

            if(bestK == -1 || (l > 0 && l < maxL && v < bestV)) {bestK = k; bestV = v;}
            if(l < 0 || l >= maxL) break;
        }
        DEBUG(bestV);
        //DEBUG(LSHGetL(bestK, p1, e));
        return bestK;
    }
}

template<typename HASHER> class LSH
{
    typedef typename HASHER::ITEM_TYPE ITEM;
    typedef typename HASHER::RESULT_TYPE RESULT_TYPE;
    Vector<ChainingHashTable<RESULT_TYPE, Vector<int> > > buckets;
    Vector<ITEM> items;
    HASHER g;
    double r2;
public:
    LSH(HASHER const& theG, int l, double theR2): buckets(l), g(theG), r2(theR2){}
    void insert(ITEM const& x)
    {
        for(int i = 0; i < buckets.getSize(); ++i)
        {
            typename HASHER::RESULT_TYPE hash = g(x, i);
            //DEBUG(i);
            //DEBUG(hash);
            Vector<int>* xBucket = buckets[i].find(hash);
            if(!xBucket)
            {
                buckets[i].insert(hash, Vector<int>());
                xBucket = buckets[i].find(hash);
            }
            xBucket->append(items.getSize());//have linear probing return chain instead?
        }
        items.append(x);
    }
    Vector<ITEM> cNeighbors(ITEM const& x)
    {
        Vector<ITEM> result;
        ChainingHashTable<int, bool> retrievedItems;
        int hitItems = 0;
        for(int i = 0; i < buckets.getSize(); ++i)
        {
            typename HASHER::RESULT_TYPE hash = g(x, i);
            //DEBUG(i);
            //DEBUG(hash);
            Vector<int>* xBucket = buckets[i].find(hash);
            if(xBucket)
                for(int i = 0; i < xBucket->getSize(); ++i)
                {
                    int itemIndex = (*xBucket)[i];
                    ++hitItems;
                    if(!retrievedItems.find(itemIndex))
                    {
                        retrievedItems.insert(itemIndex, true);
                        if(HASHER::distance(x, items[itemIndex]) < r2)
                            result.append(items[itemIndex]);
                    }
                }
        }
        //DEBUG(hitItems);
        return result;
    }
};

LSH<E2LSHHasher> buildE2LSH(int D, double r, double c, int maxL, double e = 10e-6, int maxN = 1000000)
{
    double p1 = E2LSHHasher::p(1, 1), r2 = r * (1 + c);
    int k = LSHKLFinder::LSHFindK(e, p1, E2LSHHasher::p(r, r2), maxN, maxL);
    //DEBUG(k);
    int l = LSHKLFinder::LSHGetL(k, p1, e);
    //DEBUG(l);
    return LSH<E2LSHHasher>(E2LSHHasher(k, l, D, r), l, r2);
}

template<typename HASHER> class NearestNeighborLSH
{
    typedef typename HASHER::ITEM_TYPE ITEM;
    Vector<LSH<HASHER> > lshs;//items are duplicated dont store them!
public:
    void addLSH(LSH<HASHER> const& lsh){lshs.append(lsh);}
    void insert(ITEM const& x)
        {for(int i = 0; i < lshs.getSize(); ++i) lshs[i].insert(x);}
    pair<ITEM, bool> cNeighbor(ITEM const& x)
    {
        for(int i = 0; i < lshs.getSize(); ++i)
        {
            Vector<ITEM> items = lshs[i].cNeighbors(x);
            if(items.getSize() > 0)
            {
                int best = -1, bestD;
                for(int j = 0; j < items.getSize(); ++j)
                {
                    double d = HASHER::distance(x, items[j]);
                    if(best == -1 || d < bestD)
                    {
                        best = j;
                        bestD = d;
                    }
                }
                return pair<ITEM, bool>(items[best], true);
            }
        }
        return pair<ITEM, bool>(ITEM(), false);
    }
};

NearestNeighborLSH<E2LSHHasher> buildE2NNLSH(int D, double rMin, double rMax, int maxL, double c = 1, double e = 10e-6, int maxN = 1000000)
{
    NearestNeighborLSH<E2LSHHasher> result;
    for(double r = rMin; r < rMax; r *= (1 + c))
    {
        result.addLSH(buildE2LSH(D, r, c, maxL, e, maxN));
    }
    return result;
}

void testLSH()
{
    //DEBUG(1/exp(1));
    //DEBUG(E2LSHHasher::p(1, 1));
    int D = 2;
    LSH<E2LSHHasher> tree = buildE2LSH(D, 1, 1, D * 5);
    int N = 1000000;
    for(int i = 0; i < N; ++i)
    {
        tree.insert(Vector<double>(2, i));
    }
    for(int i = 0; i < 1; ++i)
    {
        Vector<Vector<double> > neighbors = tree.cNeighbors(Vector<double>(2, i));
        DEBUG(neighbors.getSize());
        for(int j = 0; j < neighbors.getSize(); ++j)
        {
            DEBUG(j);
            for(int k = 0; k < neighbors[j].getSize(); ++k) DEBUG(neighbors[j][k]);
        }
    }
}

void testLSH2()
{
    //DEBUG(1/exp(1));
    //DEBUG(E2LSHHasher::p(1, 1));
    int D = 2;
    NearestNeighborLSH<E2LSHHasher> tree = buildE2NNLSH(D, 1, 10, D * 5);
    int N = 100000;
    for(int i = 0; i < N; ++i)
    {
        tree.insert(Vector<double>(2, i));
    }
    int noneCount = 0;
    for(int i = 0; i < N; ++i)
    {
        pair<Vector<double>, bool> neighbor = tree.cNeighbor(Vector<double>(2, i));
        //DEBUG(neighbor.second);
        if(neighbor.second)
        {
            //for(int k = 0; k < neighbor.first.getSize(); ++k) DEBUG(neighbor.first[k]);
        }
        else ++noneCount;
    }
    DEBUG(noneCount);
}

void testLSH3()
{
    //DEBUG(1/exp(1));
    //DEBUG(E2LSHHasher::p(1, 1));
    int D = 100;
    NearestNeighborLSH<E2LSHHasher> tree = buildE2NNLSH(D, 1, 10, D * 0.5);
    int N = 100000;
    for(int i = 0; i < N; ++i)
    {
        Vector<double> x;
        for(int j = 0; j < D; ++j) x.append(i);
        tree.insert(x);
    }
    int noneCount = 0;
    for(int i = 0; i < N; ++i)
    {
        Vector<double> x;
        for(int j = 0; j < D; ++j) x.append(i);
        pair<Vector<double>, bool> neighbor = tree.cNeighbor(x);
        //DEBUG(neighbor.second);
        if(neighbor.second)
        {
            //for(int k = 0; k < neighbor.first.getSize(); ++k) DEBUG(neighbor.first[k]);
        }
        else ++noneCount;
    }//20 secs
    DEBUG(noneCount);
}

void testKD3()
{
    KDTree<Point<double, 100>, bool> kdtree(100);
    int D = 100;
    int N = 100000;
    for(int i = 0; i < N; ++i)
    {
        Point<double, 100> x;
        for(int j = 0; j < D; ++j) x[j] = j;
        kdtree.insert(x, true);
    }
    for(int i = 0; i < N; ++i)
    {
        Point<double, 100> x;
        for(int j = 0; j < D; ++j) x[j] = j + GlobalRNG().uniform01();
        assert(kdtree.nearestNeighbor(x, EuclideanDistance<Point<double, 100> >::DistanceIncremental()));
    }
}
//LSH not better then KD-tree even for ML because index memory needed > data memory? Though much better than brute force!
//Before release or use must find data set or use case where it's much better!
void testKNNBF()
{
    KNNBruteForce<Point<double, 100>, bool, EuclideanDistance<Point<double, 100> >::Distance> kdtree;
    int D = 100;
    int N = 100000;
    for(int i = 0; i < N; ++i)
    {
        Point<double, 100> x;
        for(int j = 0; j < D; ++j) x[j] = j;
        kdtree.insert(x, true);
    }
    for(int i = 0; i < N; ++i)
    {
        Point<double, 100> x;
        for(int j = 0; j < D; ++j) x[j] = j + GlobalRNG().uniform01();
        assert(kdtree.nearestNeighbor(x));
    }
}

void DDDVPTree()
{
    Random<> rng(0);
    VpTree<Point<double>, int, EuclideanDistance<Point<double> >::DistanceIncremental> VPTree0to9;
    int D = 2;
    for(int i = 0; i < 10; ++i)
    {
        Point<double, 2> x;
        for(int j = 0; j < D; ++j) x[j] = rng.uniform01();
        VPTree0to9.insert(x, i);
    }

    cout << "breakpoint" << endl;
}

void DDDKDTree()
{
    Random<> rng(0);
    KDTree<Point<double, 2>, int> KDTree0to9(2);
    int D = 2;
    for(int i = 0; i < 10; ++i)
    {
        Point<double, 2> x;
        for(int j = 0; j < D; ++j) x[j] = rng.uniform01();
        KDTree0to9.insert(x, i);
    }

    cout << "breakpoint" << endl;
}

int main()
{
    DDDVPTree();
    DDDKDTree();
    testAllAutoComputationalGeometry();
    /*for(double i = 1; i <= 8; i++)
    {
        double p1 = E2LSHHasher::p(1, i);
        double p2 = E2LSHHasher::p(1.5, i);
        DEBUG(i);
        DEBUG(p1);
        DEBUG(p2);
        int l = 50;
        int k = 20;
        //for(int k = 1; k <= l; ++k)
        {
            //LSHKLFinder::LSHGetL(k, p1, 0.1);
            //if(l > 100) break;
            double p1Real = 1 - pow((1 - pow(p1, k)), l);
            double p2Real = 1 - pow((1 - pow(p2, k)), l);
            //if(p1Real < 0.9 || p2Real > 0.5) continue;
            DEBUG(k);
            DEBUG(l);
            DEBUG(p1Real);
            DEBUG(p2Real);
        }
        system("PAUSE");
    }*/
    /*testLSH();
    testLSH2();
    testLSH3();*/
    /*testKD3();//very fast
    //testKNNBF();//very slow*/
	return 0;
}
