#ifndef REGRESSION_H
#define REGRESSION_H
#include "LearningCommon.h"
#include "../Utils/Utils.h"
#include "../MiscAlgs/Misc.h"
#include "../HashTable/LinearProbingHashTable.h"
#include "../ComputationalGeometry/KDTree.h"
#include "../ComputationalGeometry/Point.h"
#include "../NumericalMethods/Matrix.h"
#include "../RandomTreap/LCPTreap.h"
#include "../RandomNumberGeneration/Statistics.h"
#include <cmath>

namespace igmdk{

struct RegressionStats
{
    double expStd, rmse, l1Err, lInfErr;
    void debug()const
    {
        DEBUG(expStd);
        DEBUG(rmse);
        DEBUG(l1Err);
        DEBUG(lInfErr);
    }
};
RegressionStats evaluateRegressor(
    Vector<pair<double, double> > const& testResult)
{
    IncrementalStatistics yStats, l2Stats, l1Stats;
    for(int i = 0; i < testResult.getSize(); ++i)
    {
        yStats.addValue(testResult[i].first);
        double diff = testResult[i].second - testResult[i].first;
        l1Stats.addValue(abs(diff));
        l2Stats.addValue(diff * diff);
    }
    RegressionStats result;
    result.lInfErr = l1Stats.maximum;
    result.l1Err = l1Stats.getMean();
    result.rmse = sqrt(l2Stats.getMean());
    result.expStd = 1 - result.rmse/yStats.stdev();
    return result;
}
template<typename LEARNER, typename DATA, typename PARAMS> double
    crossValidateReg(PARAMS const& p, DATA const& data, int nFolds = 5)
{
    return evaluateRegressor(crossValidateGeneral<LEARNER,
        typename DATA::Y_TYPE>(p, data, nFolds)).rmse;
}
template<typename LEARNER, typename PARAM, typename DATA>
struct RRiskFunctor
{
    DATA const& data;
    RRiskFunctor(DATA const& theData): data(theData) {}
    double operator()(PARAM const& p)const
        {return crossValidateReg<LEARNER>(p, data);}
};

template<typename LEARNER, typename DATA, typename PARAMS> double
    repeatedCVReg(PARAMS const& p, DATA const& data, int nFolds = 5,
    int nRepeats = 5)
{
    return evaluateRegressor(repeatedCVGeneral<double>(
        LEARNER(data, p), data, nFolds, nRepeats)).rmse;
}
template<typename LEARNER, typename PARAM, typename DATA>
struct RRCVRiskFunctor
{
    DATA const& data;
    RRCVRiskFunctor(DATA const& theData): data(theData) {}
    double operator()(PARAM const& p)const
        {return repeatedCVReg<LEARNER>(p, data);}
};

class L1LinearReg
{
    Vector<double> w;
    double b, l, r;
    int learnedCount;//this and r are only for online learning with SGD
    double f(NUMERIC_X const& x)const{return dotProduct(w, x) + b;}
    template<typename DATA> void coordinateDescent(DATA const& data,
        int maxIterations, double eps)
    {
        assert(data.getSize() > 0);
        int D = getD(data);
        Vector<double> sums(data.getSize());
        for(int i = 0; i < data.getSize(); ++i) sums[i] = data.getY(i);
        bool done = false;
        while(!done && maxIterations--)
        {
            done = true;
            for(int j = -1; j < D; ++j)
            {
                double oldVar = j == -1 ? b : w[j];
                //remove current var from sum
                for(int i = 0; i < data.getSize(); ++i)
                    sums[i] += j == -1 ? b : w[j] * data.getX(i, j);
                //solve for opt current var
                if(j == -1)
                {//update bias
                    IncrementalStatistics s;
                    for(int i = 0; i < data.getSize(); ++i)
                        s.addValue(sums[i]);
                    b = s.getMean();
                }
                else
                {//update weight
                    double a = 0, c = 0;
                    for(int i = 0; i < data.getSize(); ++i)
                    {
                        double xij = data.getX(i, j);
                        a += sums[i] * xij;
                        c += l * xij * xij;
                    }
                    if(a < -l) w[j] = (a - l)/c;
                    else if(a > l) w[j] = (a + l)/c;
                    else w[j] = 0;
                }
                //add back current var to up
                for(int i = 0; i < data.getSize(); ++i)
                    sums[i] -= j == -1 ? b : w[j] * data.getX(i, j);
                if(abs((j == -1 ? b : w[j]) - oldVar) > eps) done = false;
            }
        }
    }
public:
    template<typename DATA> L1LinearReg(DATA const& data, double theL,
        int nCoord = 1000): l(theL/2), b(0), w(getD(data)), learnedCount(-1)
        {coordinateDescent(data, nCoord, pow(10, -6));}
    typedef pair<int, pair<double, double> > PARAM;//D/l/r
    L1LinearReg(PARAM const& p): l(p.second.first/2), r(p.second.second),
        b(0), w(p.first), learnedCount(0) {}
    void learn(NUMERIC_X const& x, double y)
    {
        assert(learnedCount != -1);//can't mix batch and offline
        double rate = r * RMRate(learnedCount++), err = y - f(x);
        //l/n*|w| + (y - wx + b)2
        //dw = l/n*sign(w) - x(y - (wx + b));
        //db = - (y - (wx + b))
        for(int i = 0; i < w.getSize(); ++i) w[i] +=
            rate * (x[i] * err - (w[i] > 0 ? 1 : -1) * l/learnedCount);
        b += rate * err;
    }
    double predict(NUMERIC_X const& x)const{return f(x);}
    template<typename MODEL, typename DATA>
    static double findL(DATA const& data)
    {
        int lLow = -15, lHigh = 5;
        Vector<double> regs;
        for(double j = lHigh; j > lLow; j -= 2) regs.append(pow(2, j));
        return valMinFunc(regs.getArray(), regs.getSize(),
            RRiskFunctor<MODEL, double, DATA>(data));
    }
};
struct NoParamsL1LinearReg
{
    L1LinearReg model;
    template<typename DATA> NoParamsL1LinearReg(DATA const& data):
        model(data, L1LinearReg::findL<L1LinearReg>(data)) {}
    double predict(NUMERIC_X const& x)const{return model.predict(x);}
};
typedef ScaledLearner<NoParamsLearner<NoParamsL1LinearReg, double>, double>
    SLasso;

class SRaceLasso
{
    ScalerMQ s;
    RaceLearner<L1LinearReg, L1LinearReg::PARAM> model;
    static Vector<L1LinearReg::PARAM> makeParams(int D)
    {
        Vector<L1LinearReg::PARAM> result;
        int lLow = -15, lHigh = 5, rLow = -15, rHigh = 5;
        for(int j = lHigh; j > lLow; j -= 2)
        {
            double l = pow(2, j);
            for(int i = rHigh; i > rLow; i -= 2) result.append(
                L1LinearReg::PARAM(D, pair<double, double>(l, pow(2, i))));
        }
        return result;
    }
public:
    template<typename DATA> SRaceLasso(DATA const& data):
        model(makeParams(getD(data))), s(getD(data))
    {
        for(int j = 0; j < 1000000; ++j)
        {
            int i = GlobalRNG().mod(data.getSize());
            learn(data.getX(i), data.getY(i));
        }
    }
    SRaceLasso(int D): model(makeParams(D)), s(D) {}
    void learn(NUMERIC_X const& x, double y)
    {
        s.addSample(x);
        model.learn(s.scale(x), y);
    }
    double predict(NUMERIC_X const& x)const
        {return model.predict(s.scale(x));}
};

struct RegressionTree
{
    struct Node
    {
        union
        {
            double split;//for internal nodes
            double label;//for leaf nodes
        };
        int feature;//for internal nodes
        Node *left, *right;
        bool isLeaf(){return !left;}
        Node(int theFeature, double theSplit): feature(theFeature),
            split(theSplit), left(0), right(0) {}
    }* root;
    Freelist<Node> f;
    double SSE(double sum, double sum2, int n)const
        {return sum2 - sum * sum/n;}
    template<typename DATA> struct Comparator
    {
        int feature;
        DATA const& data;
        double v(int i)const{return data.data.getX(i, feature);}
        bool operator()(int lhs, int rhs)const{return v(lhs) < v(rhs);}
        bool isEqual(int lhs, int rhs)const{return v(lhs) == v(rhs);}
    };
    void rDelete(Node* node)
    {
        if(node)
        {
            rDelete(node->left);
            f.remove(node->left);
            rDelete(node->right);
            f.remove(node->right);
        }
    }
    double classifyHelper(NUMERIC_X const& x, Node* current)const
    {
        while(!current->isLeaf()) current = x[current->feature] <
            current->split ? current->left : current->right;
        return current->label;
    }
    template<typename DATA> Node* rHelper(DATA& data, int left, int right,
        double pruneZ, int depth, bool rfMode)
    {
        int D = data.getX(left).getSize(), bestFeature = -1,
            n = right - left + 1;
        double bestSplit, bestScore, sumY = 0, sumY2 = 0;
        Comparator<DATA> co = {-1, data};
        for(int j = left; j <= right; ++j)
        {
            double y = data.getY(j);
            sumY += y;
            sumY2 += y * y;
        }
        double ave = sumY/n, sse = max(0.0, SSE(sumY, sumY2, n));
        Bitset<> allowedFeatures;
        if(rfMode)
        {//sample features for random forest
            allowedFeatures = Bitset<>(D);
            allowedFeatures.setAll(0);
            Vector<int> p = GlobalRNG().sortedSample(sqrt(D), D);
            for(int j = 0; j < p.getSize(); ++j)allowedFeatures.set(p[j], 1);
        }
        if(sse > 0) for(int i = 0; i < D; ++i)//find best feature and split
            if(allowedFeatures.getSize() == 0 || allowedFeatures[i])
            {
                co.feature = i;
                quickSort(data.permutation.getArray(), left, right, co);
                double sumYLeft = 0, sumYRight = sumY, sumY2Left = 0,
                    sumY2Right = sumY2;
                int nRight = n, nLeft = 0;
                for(int j = left; j < right; ++j)
                {//incrementally roll counts
                    int y = data.getY(j);
                    ++nLeft;
                    sumYLeft += y;
                    sumY2Left += y * y;
                    --nRight;
                    sumYRight -= y;
                    sumY2Right -= y * y;
                    double fLeft = data.getX(j, i), score =
                        SSE(sumYLeft, sumY2Left, nLeft) +
                        SSE(sumYRight, sumY2Right, nRight),
                        fRight = data.getX(j + 1, i);
                    if(fLeft != fRight && //don't split equal values
                        (bestFeature == -1 || score < bestScore))
                    {
                        bestScore = score;
                        bestSplit = (fLeft + fRight)/2;
                        bestFeature = i;
                    }
                }
            }
        if(n < 3 || depth <= 1 || sse <= 0 || bestFeature == -1)
            return new(f.allocate())Node(-1, ave);
        //split examples into left and right
        int i = left - 1;
        for(int j = left; j <= right; ++j)
            if(data.getX(j, bestFeature) < bestSplit)
                swap(data.permutation[j], data.permutation[++i]);
        if(i < left || i > right) return new(f.allocate())Node(-1, ave);
        Node* node = new(f.allocate())Node(bestFeature, bestSplit);
        //recursively compute children
        node->left = rHelper(data, left, i, pruneZ, depth - 1, rfMode);
        node->right = rHelper(data, i + 1, right, pruneZ, depth - 1, rfMode);
        //try to prune
        double nodeWins = 0, treeWins = 0;
        for(int j = left; j <= right; ++j)
        {
            double y = data.getY(j), eNode = ave - y, eTree =
                classifyHelper(data.getX(j), node) - y;
            if(eNode * eNode == eTree * eTree)
            {
                nodeWins += 0.5;
                treeWins += 0.5;
            }
            else if(eNode * eNode < eTree * eTree) ++nodeWins;
            else ++treeWins;
        }
        if(!rfMode && signTestAreEqual(nodeWins, treeWins, pruneZ))
        {
            rDelete(node);
            node->left = node->right = 0;
            node->label = ave;
            node->feature = -1;
        }
        return node;
    }
    Node* constructFrom(Node* node)
    {
        Node* tree = 0;
        if(node)
        {
            tree = new(f.allocate())Node(*node);
            tree->left = constructFrom(node->left);
            tree->right = constructFrom(node->right);
        }
        return tree;
    }
public:
    template<typename DATA> RegressionTree(DATA const& data, double pruneZ =
        0.25, int maxDepth = 50, bool rfMode = false): root(0)
    {
        assert(data.getSize() > 0);
        int left = 0, right = data.getSize() - 1;
        PermutedData<DATA> pData(data);
        for(int i = 0; i < data.getSize(); ++i) pData.addIndex(i);
        root = rHelper(pData, left, right, pruneZ, maxDepth, rfMode);
    }
    RegressionTree(RegressionTree const& other)
        {root = constructFrom(other.root);}
    RegressionTree& operator=(RegressionTree const& rhs)
        {return genericAssign(*this, rhs);}
    double predict(NUMERIC_X const& x)const
        {return root ? classifyHelper(x, root) : 0;}
};

class RandomForestReg
{
    Vector<RegressionTree> forest;
public:
    template<typename DATA> RandomForestReg(DATA const& data,
        int nTrees = 300){addTrees(data, nTrees);}
    template<typename DATA> void addTrees(DATA const& data, int nTrees)
    {
        assert(data.getSize() > 1);
        for(int i = 0, D = getD(data); i < nTrees; ++i)
        {
            PermutedData<DATA> resample(data);
            for(int j = 0; j < data.getSize(); ++j)
                resample.addIndex(GlobalRNG().mod(data.getSize()));
            forest.append(RegressionTree(resample, 0, 50, true));
        }
    }
    double predict(NUMERIC_X const& x)const
    {
        IncrementalStatistics s;
        for(int i = 0; i < forest.getSize(); ++i)
            s.addValue(forest[i].predict(x));
        return s.getMean();
    }
};

template<typename X = NUMERIC_X, typename INDEX = VpTree<X, double, typename
    EuclideanDistance<X>::Distance> > class KNNReg
{
    mutable INDEX instances;
    int k;
public:
    template<typename DATA> KNNReg(DATA const& data, int theK = -1): k(theK)
    {
        assert(data.getSize() > 0);
        if(k == -1) k = 2 * int(log(data.getSize())/2) + 1;
        for(int i = 0; i < data.getSize(); ++i)
            learn(data.getY(i), data.getX(i));
    }
    void learn(double label, X const& x){instances.insert(x, label);}
    double predict(X const& x)const
    {
        Vector<typename INDEX::NodeType*> neighbors = instances.kNN(x, k);
        IncrementalStatistics s;
        for(int i = 0; i < neighbors.getSize(); ++i)
            s.addValue(neighbors[i]->value);
        return s.getMean();
    }
};
typedef ScaledLearner<NoParamsLearner<KNNReg<>, double>, double> SKNNReg;

class HiddenLayerNNReg
{
    Vector<NeuralNetwork> nns;
public:
    template<typename DATA> HiddenLayerNNReg(DATA const& data,
        Vector<double>const& p, int nGoal = 100000, int nNns = 5):
        nns(nNns, NeuralNetwork(getD(data), true, p[0]))
    {//structure
        int nHidden = p[1], D = getD(data),
            nRepeats = ceiling(nGoal, data.getSize());
        double a = sqrt(3.0/D);
        for(int l = 0; l < nns.getSize(); ++l)
        {
            NeuralNetwork& nn = nns[l];
            nn.addLayer(nHidden);
            for(int j = 0; j < nHidden; ++j)
                for(int k = -1; k < D; ++k)
                    nn.addConnection(0, j, k, k == -1 ? 0 :
                        GlobalRNG().uniform(-a, a));
            nn.addLayer(1);
            for(int k = -1; k < nHidden; ++k)
                nn.addConnection(1, 0, k, 0);
        }
        //training
        for(int j = 0; j < nRepeats; ++j)
            for(int i = 0; i < data.getSize(); ++i)
                learn(data.getX(i), data.getY(i));
    }
    void learn(NUMERIC_X const& x, double label)
    {
        for(int l = 0; l < nns.getSize(); ++l)
            nns[l].learn(x, Vector<double>(1, label));
    }
    double evaluate(NUMERIC_X const& x)const
    {
        double result = 0;
        for(int l = 0; l < nns.getSize(); ++l)
            result += nns[l].evaluate(x)[0];
        return result/nns.getSize();
    }
    int predict(NUMERIC_X const& x)const{return evaluate(x);}
};
struct NoParamsNNReg
{
    HiddenLayerNNReg model;
    template<typename DATA> static Vector<double> findParams(DATA const&
        data, int rLow = -15, int rHigh = 5, int hLow = 0, int hHigh = 6)
    {
        Vector<Vector<double> > sets(2);
        for(int i = rLow; i <= rHigh; i += 2) sets[0].append(pow(2, i));
        for(int i = hLow; i <= hHigh; i += 2) sets[1].append(pow(2, i));
        return gridMinimize(sets,
            RRiskFunctor<HiddenLayerNNReg, Vector<double>, DATA>(data));
    }
    template<typename DATA> NoParamsNNReg(DATA const& data):
        model(data, findParams(data)) {}
    double predict(NUMERIC_X const& x)const{return model.predict(x);}
};
typedef ScaledLearner<NoParamsLearner<NoParamsNNReg, double>, double, EMPTY,
    ScalerMQ> SNNReg;

template<typename SUBSET_LEARNER = RandomForestReg> struct SmartFSLearnerReg
{
    typedef FeatureSubsetLearner<SUBSET_LEARNER> MODEL;
    MODEL model;
public:
    template<typename DATA> SmartFSLearnerReg(DATA const& data,
        int subsampleLimit = 20): model(data, selectFeaturesSmart(
        RRiskFunctor<MODEL, Bitset<>, DATA>(data), getD(data),
        subsampleLimit)) {}
    double predict(NUMERIC_X const& x)const{return model.predict(x);}
};

class SimpleBestCombinerReg
{
    BestCombiner<double> c;
public:
    template<typename DATA> SimpleBestCombinerReg(DATA const& data)
    {
        c.addNoParamsClassifier<RandomForestReg>(data, RRiskFunctor<
            NoParamsLearner<RandomForestReg, double>, EMPTY, DATA>(data));
        c.addNoParamsClassifier<SLasso>(data, RRiskFunctor<
            NoParamsLearner<SLasso, double>, EMPTY, DATA>(data));
    }
    double predict(NUMERIC_X const& x)const{return c.predict(x);}
};


}//end namespace
#endif

