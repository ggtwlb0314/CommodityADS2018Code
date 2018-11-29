#ifndef MACHINELEARNING_H
#define MACHINELEARNING_H
#include "LearningCommon.h"
#include "../Utils/Utils.h"
#include "../MiscAlgs/Misc.h"
#include "../HashTable/LinearProbingHashTable.h"
#include "../ComputationalGeometry/KDTree.h"
#include "../ComputationalGeometry/Point.h"
#include "../NumericalMethods/Matrix.h"
#include "../NumericalMethods/NumericalMethods.h"
#include "../NumericalMethods/NumericalOptimization.h"
#include "../RandomTreap/LCPTreap.h"
#include "../RandomNumberGeneration/Statistics.h"
#include <cmath>
namespace igmdk{

template<typename DATA> int findNClasses(DATA const& data)
{
    int maxClass = -1;
    for(int i = 0; i < data.getSize(); ++i)
        maxClass = max(maxClass, data.getY(i));
    return maxClass + 1;
}

template<typename DATA> pair<PermutedData<DATA>, PermutedData<DATA> >
    createTrainingTestSetsStatified(DATA const& data,
    double relativeTestSize = 0.8)
{
    int n = data.getSize(), m = n * relativeTestSize;
    assert(m > 0 && m < n);
    pair<PermutedData<DATA>, PermutedData<DATA> > result(data, data);
    Vector<int> counts(findNClasses(data)), p(n);//need p for legacy only
    for(int i = 0; i < n; ++i){++counts[data.getY(i)]; p[i] = i;}
    for(int i = 0; i < counts.getSize(); ++i) counts[i] *= relativeTestSize;
    for(int i = 0; i < p.getSize(); ++i)
    {
        int label = data.getY(p[i]);
        if(counts[label]){--counts[label]; result.first.addIndex(p[i]);}
        else
        {
            result.second.addIndex(p[i]);
            p[i--] = p.lastItem();
            p.removeLast();
        }
    }
    return result;
}

Matrix<int> evaluateConfusion(Vector<pair<int, int> > const& testResult,
    int nClasses = -1)
{
    if(nClasses == -1)
    {//calculate nClasses if unknown
        int maxClass = 0;
        for(int i = 0; i < testResult.getSize(); ++i) maxClass =
            max(maxClass, max(testResult[i].first, testResult[i].second));
        nClasses = maxClass + 1;
    }
    Matrix<int> result(nClasses, nClasses);
    for(int i = 0; i < testResult.getSize(); ++i)
        ++result(testResult[i].first, testResult[i].second);
    return result;
}

double evalConfusionCost(Matrix<int>const& confusion,
    Matrix<double>const& cost)
{
    int k = confusion.rows, total = 0;
    assert(k == confusion.columns && k == cost.rows && k == cost.columns);
    double sum = 0;
    for(int r = 0; r < k; ++r)
        for(int c = 0; c < k; ++c)
        {
            total += confusion(r, c);
            sum += confusion(r, c) * cost(r, c);
        }
    return sum/total;
}

struct ClassifierStats
{
    struct Bound
    {
        double mean, lower, upper;
        Bound(): mean(0), lower(0), upper(0) {}
        Bound(double theMean, pair<double, double> const& bounds):
            mean(theMean), lower(bounds.first), upper(bounds.second) {}
        Bound operator*=(double a)
        {
            mean *= a;
            lower *= a;
            upper *= a;
        }
    };
    Bound acc, bac;
    Vector<Bound> accByClass, confByClass;
    int total;
    static Bound getBound(IncrementalStatistics s)
    {
        double inf = numeric_limits<double>::infinity();
        pair<double, double> bound(-inf, inf);
        double mean = inf;
        if(s.n > 0)
        {
            mean = s.getMean();
            bound = s.bernBounds(0.95);
        }
        return Bound(mean, bound);
    }
    ClassifierStats(Matrix<int> const& confusion): total(0)
    {//same row = same label, same column = same prediction
        Vector<int> confTotal, accTotal;
        int k = confusion.getRows(), nBac = 0, actualK = 0;
        IncrementalStatistics accS, basSW;
        Vector<IncrementalStatistics> precS(k);
        Vector<double> weights(k);
        for(int r = 0; r < k; ++r)
        {
            int totalR = 0;
            for(int c = 0; c < k; ++c)
            {
                totalR += confusion(r, c);
                weights[r] += confusion(r, c);
                total += confusion(r, c);
            }
            accTotal.append(totalR);
            actualK += (totalR > 0);
        }
        double M = 0;
        for(int r = 0; r < k; ++r)
        {
            weights[r] = total/weights[r]/actualK;
            IncrementalStatistics bacS;
            for(int c = 0; c < k; ++c)
            {
                int count = confusion(r, c);
                bool correct = r == c;
                while(count--)
                {
                    accS.addValue(correct);
                    basSW.addValue(correct * weights[r]);
                    M += weights[r] * weights[r];
                    bacS.addValue(correct);
                    precS[c].addValue(correct);
                }
            }
            accByClass.append(getBound(bacS));
        }
        M = sqrt(M/total);
        for(int c = 0; c < k; ++c)
        {
            int totalC = 0;
            for(int r = 0; r < k; ++r) totalC += confusion(r, c);
            confTotal.append(totalC);
            confByClass.append(getBound(precS[c]));
        }
        acc = getBound(accS);
        bac = getBound(basSW);
        bac.lower *= M;
        bac.upper *= M;
    }
    void debug()const
    {
        DEBUG(acc.mean * total);
        DEBUG(total);
        cout << "Accuracy: ";
        cout << acc.mean << " ";
        cout << "95% Bernstein range: ";
        cout << acc.lower << " ";
        cout << acc.upper;
        cout << endl;
        cout << "Balanced Accuracy: ";
        cout << bac.mean << " ";
        cout << "95% Bernstein range: ";
        cout << bac.lower << " ";
        cout << bac.upper;
        cout << endl;
        cout << "Accuracy by class and 95% range: " << endl;
        for(int i = 0; i < accByClass.getSize(); ++i)
        {
            cout << accByClass[i].mean << " ";
            cout << accByClass[i].lower << " ";
            cout << accByClass[i].upper;
            cout << endl;
        }
        cout << "Confidence by class and 95% range: " << endl;
        for(int i = 0; i < confByClass.getSize(); ++i)
        {
            cout << confByClass[i].mean << " ";
            cout << confByClass[i].lower << " ";
            cout << confByClass[i].upper;
            cout << endl;
        }
    }
};

template<typename LEARNER, typename DATA, typename PARAMS>
    Vector<pair<int, int> > crossValidationStratified(PARAMS const& p,
    DATA const& data, int nFolds = 5)
{
    assert(nFolds > 1 && nFolds <= data.getSize());
    int nClasses = findNClasses(data), testSize = 0;
    Vector<int> counts(nClasses, 0), starts(nClasses, 0);
    PermutedData<DATA> pData(data);
    for(int i = 0; i < data.getSize(); ++i)
    {
        pData.addIndex(i);
        ++counts[data.getY(i)];
    }
    for(int i = 0; i < counts.getSize(); ++i)
        counts[i] /= nFolds;//roundoff goes to training
    for(int i = 0; i < counts.getSize(); ++i) testSize += counts[i];
    Vector<pair<int, int> > result;
    for(int i = 0;; ++i)
    {//create list of included test examples in increasing order
        Vector<int> includedCounts(nClasses, 0), includedIndices;
        for(int j = valMin(starts.getArray(), starts.getSize());
            includedIndices.getSize() < testSize; ++j)
        {
            int label = data.getY(j);
            if(starts[label] <= j && includedCounts[label] < counts[label])
            {
                ++includedCounts[label];
                includedIndices.append(j);
                starts[label] = j + 1;
            }
        }
        PermutedData<DATA> testData(data);
        for(int j = testSize - 1; j >= 0; --j)
        {
            testData.addIndex(includedIndices[j]);
            pData.permutation[includedIndices[j]] =
                pData.permutation.lastItem();
            pData.permutation.removeLast();
        }
        result.appendVector(evaluateLearner<int>(LEARNER(pData, p),
            testData));
        //put test data back into data in correct places
        if(i == nFolds - 1) break;
        for(int j = 0; j < testSize; ++j)
        {
            pData.addIndex(includedIndices[j]);
            pData.permutation[includedIndices[j]] =
                testData.permutation[testSize - 1 - j];
        }
    }
    return result;
}
template<typename LEARNER, typename DATA, typename PARAMS> double
    crossValidation(PARAMS const& p, DATA const& data, int nFolds = 5)
{
    return ClassifierStats(evaluateConfusion(
        crossValidationStratified<LEARNER>(p, data, nFolds))).acc.mean;
}

template<typename LEARNER, typename PARAM, typename DATA>
struct SCVRiskFunctor
{
    DATA const& data;
    SCVRiskFunctor(DATA const& theData): data(theData) {}
    double operator()(PARAM const& p)const
        {return 1 - crossValidation<LEARNER>(p, data);}
};

struct DecisionTree
{
    struct Node
    {
        union
        {
            int feature;//for internal nodes
            int label;//for leaf nodes
        };
        double split;
        Node *left, *right;
        bool isLeaf(){return !left;}
        Node(int theFeature, double theSplit): feature(theFeature),
            split(theSplit), left(0), right(0) {}
    }*root;
    Freelist<Node> f;
    double H(double p){return p > 0 ? p * log(1/p) : 0;}
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
    typedef pair<Node*, int> RTYPE;
    template<typename DATA> RTYPE rHelper(DATA& data, int left, int right,
        int nClasses, double pruneZ, int depth, bool rfMode)
    {
        int D = data.getX(left).getSize(), bestFeature = -1,
            n = right - left + 1;
        double bestSplit, bestRem, h = 0;
        Comparator<DATA> co = {-1, data};
        Vector<int> counts(nClasses, 0);
        for(int j = left; j <= right; ++j) ++counts[data.getY(j)];
        for(int j = 0; j < nClasses; ++j) h += H(counts[j] * 1.0/n);
        int majority = argMax(counts.getArray(), nClasses),
            nodeAccuracy = counts[majority];
        Bitset<> allowedFeatures;
        if(rfMode)
        {//sample features for random forest
            allowedFeatures = Bitset<>(D);
            allowedFeatures.setAll(0);
            Vector<int> p = GlobalRNG().sortedSample(sqrt(D), D);
            for(int j = 0; j < p.getSize(); ++j)allowedFeatures.set(p[j], 1);
        }
        if(h > 0) for(int i = 0; i < D; ++i)//find best feature and split
            if(allowedFeatures.getSize() == 0 || allowedFeatures[i])
            {
                co.feature = i;
                quickSort(data.permutation.getArray(), left, right, co);
                int nRight = n, nLeft = 0;
                Vector<int> countsLeft(nClasses, 0), countsRight = counts;
                for(int j = left; j < right; ++j)
                {//incrementally roll counts
                    int label = data.getY(j);
                    ++nLeft;
                    ++countsLeft[label];
                    --nRight;
                    --countsRight[label];
                    double fLeft = data.getX(j, i), hLeft = 0,
                        fRight = data.getX(j + 1, i), hRight = 0;
                    if(fLeft != fRight)
                    {//don't split equal values
                        for(int l = 0; l < nClasses; ++l)
                        {
                            hLeft += H(countsLeft[l] * 1.0/nLeft);
                            hRight += H(countsRight[l] * 1.0/nRight);
                        }
                        double rem = hLeft * nLeft + hRight * nRight;
                        if(bestFeature == -1 || rem < bestRem)
                        {
                            bestRem = rem;
                            bestSplit = (fLeft + fRight)/2;
                            bestFeature = i;
                        }
                    }
                }
            }
        if(depth <= 1 || h == 0 || bestFeature == -1)
            return RTYPE(new(f.allocate())Node(majority, 0), nodeAccuracy);
        //split examples into left and right
        int i = left - 1;
        for(int j = left; j <= right; ++j)
            if(data.getX(j, bestFeature) < bestSplit)
                swap(data.permutation[j], data.permutation[++i]);
        if(i < left || i > right)
            return RTYPE(new(f.allocate())Node(majority, 0), nodeAccuracy);
        Node* node = new(f.allocate())Node(bestFeature, bestSplit);
        //recursively compute children
        RTYPE lData = rHelper(data, left, i, nClasses, pruneZ, depth - 1,
            rfMode), rData = rHelper(data, i + 1, right, nClasses, pruneZ,
            depth - 1, rfMode);
        node->left = lData.first;
        node->right = rData.first;
        int treeAccuracy = lData.second + rData.second, nTreeWins =
            treeAccuracy - nodeAccuracy, nDraws = n - nTreeWins;
        //try to prune
        if(!rfMode &&
            signTestAreEqual(nDraws/2.0, nDraws/2.0 + nTreeWins, pruneZ))
        {
            rDelete(node);
            node->left = node->right = 0;
            node->label = majority;
            node->split = 0;
            treeAccuracy = nodeAccuracy;
        }
        return RTYPE(node, treeAccuracy);
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
    template<typename DATA> DecisionTree(DATA const& data, double pruneZ = 1,
        bool rfMode = false, int maxDepth = 50): root(0)
    {
        assert(data.getSize() > 0);
        int left = 0, right = data.getSize() - 1;
        PermutedData<DATA> pData(data);
        for(int i = 0; i < data.getSize(); ++i) pData.addIndex(i);
        root = rHelper(pData, left, right, findNClasses(
            data), pruneZ, maxDepth, rfMode).first;
    }
    DecisionTree(DecisionTree const& other)
        {root = constructFrom(other.root);}
    DecisionTree& operator=(DecisionTree const& rhs)
        {return genericAssign(*this, rhs);}
    int predict(NUMERIC_X const& x)const
    {
        assert(root);//check for bad data
        Node* current = root;
        while(!current->isLeaf()) current = x[current->feature] <
            current->split ? current->left : current->right;
        return current->label;
    }
};

int costClassify(Vector<double> const& probs, Matrix<double> const& cost)
{
    int k = probs.getSize();
    assert(k == cost.getRows());
    Vector<double> losses(k);
    for(int i = 0; i < k; ++i)
        for(int j = 0; j < k; ++j) losses[i] += probs[j] * cost(j, i);
    return argMin(losses.getArray(), k);
}

class RandomForest
{
    Vector<DecisionTree> forest;
    int nClasses;
public:
    template<typename DATA> RandomForest(DATA const& data, int nTrees = 300):
        nClasses(findNClasses(data))
    {
        assert(data.getSize() > 1);
        for(int i = 0; i < nTrees; ++i)
        {
            PermutedData<DATA> resample(data);
            for(int j = 0; j < data.getSize(); ++j)
                resample.addIndex(GlobalRNG().mod(data.getSize()));
            forest.append(DecisionTree(resample, 0, true));
        }
    }
    template <typename ENSEMBLE> static int classifyWork(NUMERIC_X const& x,
        ENSEMBLE const& e, int nClasses)
    {
        Vector<int> counts(nClasses, 0);
        for(int i = 0; i < e.getSize(); ++i) ++counts[e[i].predict(x)];
        return argMax(counts.getArray(), counts.getSize());
    }
    int predict(NUMERIC_X const& x)const
        {return classifyWork(x, forest, nClasses);}
    Vector<double> classifyProbs(NUMERIC_X const& x)const
    {
        Vector<double> counts(nClasses, 0);
        for(int i = 0; i < forest.getSize(); ++i)
            ++counts[forest[i].predict(x)];
        normalizeProbs(counts);
        return counts;
    }
};

class WeightedRF
{
    Vector<DecisionTree> forest;
    int nClasses;
public:
    template<typename DATA> WeightedRF(DATA const& data, Vector<double> const
        & weights, int nTrees = 300): nClasses(findNClasses(data))
    {
        assert(data.getSize() > 1);
        AliasMethod sampler(weights);
        for(int i = 0; i < nTrees; ++i)
        {
            PermutedData<DATA> resample(data);
            for(int j = 0; j < data.getSize(); ++j)
                resample.addIndex(sampler.next());
            forest.append(DecisionTree(resample, 0, true));
        }
    }
    int predict(NUMERIC_X const& x)const
        {return RandomForest::classifyWork(x, forest, nClasses);}
};

template<typename DATA>
Vector<double> findImbalanceWeights(DATA const& data)
{
    int n = data.getSize(), properK = 0, nClasses = findNClasses(data);
    Vector<double> counts(nClasses);
    for(int i = 0; i < n; ++i) ++counts[data.getY(i)];
    for(int i = 0; i < nClasses; ++i) if(counts[i] > 0) ++properK;
    Vector<double> dataWeights(n, 0);
    for(int i = 0; i < data.getSize(); ++i)
        dataWeights[i] = 1.0/properK/counts[data.getY(i)];
    return dataWeights;
}
class ImbalanceRF
{
    WeightedRF model;
public:
    template<typename DATA> ImbalanceRF(DATA const& data, int nTrees = 300):
        model(data, findImbalanceWeights(data), nTrees) {}
    int predict(NUMERIC_X const& x)const{return model.predict(x);}
};

template<typename LEARNER, typename PARAMS = EMPTY>
    class WeightedBaggedLearner
{
    Vector<LEARNER> models;
    int nClasses;
public:
    template<typename DATA> WeightedBaggedLearner(DATA const& data,
        Vector<double> weights, PARAMS const& p = PARAMS(), int nBags = 15):
        nClasses(findNClasses(data))
    {
        assert(data.getSize() > 1);
        AliasMethod sampler(weights);
        for(int i = 0; i < nBags; ++i)
        {
            PermutedData<DATA> resample(data);
            for(int j = 0; j < data.getSize(); ++j)
                resample.addIndex(sampler.next());
            models.append(LEARNER(resample, p));
        }
    }
    int predict(NUMERIC_X const& x)const
        {return RandomForest::classifyWork(x, models, nClasses);}
};

template<typename LEARNER, typename PARAMS = EMPTY, typename X = NUMERIC_X>
class OnlineMulticlassLearner
{
    mutable Treap<int, LEARNER> binaryLearners;
    int nClasses;
    PARAMS p;
    int makeKey(short label1, short label2) const
        {return label1 * numeric_limits<short>::max() + label2;}
public:
    OnlineMulticlassLearner(PARAMS const& theP = PARAMS(),
        int initialNClasses = 0): nClasses(initialNClasses), p(theP) {}
    void learn(X const& x, int label)
    {
        nClasses = max(nClasses, label + 1);
        for(int j = 0; j < nClasses; ++j) if(j != label)
        {
            int key = j < label ? makeKey(j, label) : makeKey(label, j);
            LEARNER* s = binaryLearners.find(key);
            if(!s)
            {
                binaryLearners.insert(key, LEARNER(p));
                s = binaryLearners.find(key);
            }
            s->learn(x, int(j < label));
        }
    }
    int predict(X const& x)const
    {
        assert(nClasses > 0);
        Vector<int> votes(nClasses, 0);
        for(int j = 0; j < nClasses; ++j)
            for(int k = j + 1; k < nClasses; ++k)
            {
                LEARNER* s = binaryLearners.find(makeKey(j, k));
                if(s) ++votes[s->predict(x) ? k : j];
            }
        return argMax(votes.getArray(), votes.getSize());
    }
};

template<typename LEARNER, typename PARAMS = EMPTY, typename X = NUMERIC_X>
class MulticlassLearner
{//if params not passed, uses default value!
    mutable ChainingHashTable<int, LEARNER> binaryLearners;
    int nClasses;
public:
    Vector<LEARNER const*> getLearners()const
    {
        Vector<LEARNER const*> result;
        for(typename ChainingHashTable<int, LEARNER>::Iterator i =
            binaryLearners.begin(); i != binaryLearners.end(); ++i)
            result.append(&i->value);
        return result;
    };
    template<typename DATA> MulticlassLearner(DATA const& data,
        PARAMS const&p = PARAMS()): nClasses(findNClasses(data))
    {
        Vector<Vector<int> > labelIndex(nClasses);
        for(int i = 0; i < data.getSize(); ++i)
            labelIndex[data.getY(i)].append(i);
        for(int j = 0; j < nClasses; ++j) if(labelIndex[j].getSize() > 0)
            for(int k = j + 1; k < nClasses; ++k)
                if(labelIndex[k].getSize() > 0)
                {
                    PermutedData<DATA> twoClassData(data);
                    RelabeledData<PermutedData<DATA> >
                        binaryData(twoClassData);
                    for(int l = 0, m = 0; l < labelIndex[j].getSize() ||
                        m < labelIndex[k].getSize(); ++l, ++m)
                    {
                        if(l < labelIndex[j].getSize())
                        {
                            twoClassData.addIndex(labelIndex[j][l]);
                            binaryData.addLabel(0);
                        }
                        if(m < labelIndex[k].getSize())
                        {
                            twoClassData.addIndex(labelIndex[k][m]);
                            binaryData.addLabel(1);
                        }
                    }
                    binaryLearners.insert(j * nClasses + k,
                        LEARNER(binaryData, p));
                }
    }
    int predict(X const& x)const
    {
        Vector<int> votes(nClasses, 0);
        for(int j = 0; j < nClasses; ++j)
            for(int k = j + 1; k < nClasses; ++k)
            {
                LEARNER* s = binaryLearners.find(j * nClasses + k);
                if(s) ++votes[s->predict(x) ? k : j];
            }
        return argMax(votes.getArray(), votes.getSize());
    }
    int classifyByProbs(X const& x)const
    {//for probability-output learners like neural network
        Vector<double> votes(nClasses, 0);
        for(int j = 0; j < nClasses; ++j)
            for(int k = j + 1; k < nClasses; ++k)
            {
                LEARNER* s = binaryLearners.find(j * nClasses + k);
                if(s)
                {
                    double p = s->evaluate(x);
                    votes[k] += p;
                    votes[j] += 1 - p;
                }
            }
        return argMax(votes.getArray(), votes.getSize());
    }
};

template<typename X = NUMERIC_X, typename INDEX = VpTree<X, int, typename
    EuclideanDistance<X>::Distance> > class KNNClassifier
{
    mutable INDEX instances;
    int n, nClasses;
public:
    KNNClassifier(int theNClasses): nClasses(theNClasses), n(0) {}
    template<typename DATA> KNNClassifier(DATA const& data): n(0),
        nClasses(findNClasses(data))
    {
        for(int i = 0; i < data.getSize(); ++i)
            learn(data.getY(i), data.getX(i));
    }
    void learn(int label, X const& x){instances.insert(x, label); ++n;}
    int predict(X const& x)const
    {
        Vector<typename INDEX::NodeType*> neighbors =
            instances.kNN(x, 2 * int(log(n))/2 + 1);
        Vector<int> votes(nClasses);
        for(int i = 0; i < neighbors.getSize(); ++i)
            ++votes[neighbors[i]->value];
        return argMax(votes.getArray(), votes.getSize());
    }
};

class BinaryLSVM
{
    Vector<double> w;
    double b, l;
    int learnedCount;
    static int y(bool label){return label * 2 - 1;}
    static double loss(double fxi, double yi){return max(0.0, 1 - fxi * yi);}
    double f(NUMERIC_X const& x)const{return dotProduct(w, x) + b;}
    template<typename DATA> class GSL1Functor
    {
        DATA const& data;
        mutable Vector<double> sums;
        Vector<double> &w;
        double &b, l;
        int j, D;
        mutable int evalCount;
    public:
        GSL1Functor(DATA const& theData, double& theB, Vector<double>& theW,
            double theL): data(theData), sums(theData.getSize(), theB),
            w(theW), b(theB), l(theL), j(-1), D(getD(data)), evalCount(0)
        {
            for(int i = 0; i < data.getSize(); ++i)
                for(int j = 0; j < D; ++j)
                    sums[i] += w[j] * data.getX(i, j);
        }
        void setCurrentDimension(int theJ)
        {
            assert(theJ >= 0 && theJ < D + 1);
            j = theJ;
        }
        int getEvalCount()const{return evalCount;}
        int getSize()const{return D + 1;}
        double getXi()const{return j == D ? b : w[j];}
        double operator()(double wj)const
        {
            ++evalCount;
            double result = j == D ? 0 : l * abs(wj);
            for(int i = 0; i < data.getSize(); ++i) result += loss(
                sums[i] + (j == D ? (wj - b) : (wj - w[j]) * data.getX(i, j)),
                y(data.getY(i)));
            return result/data.getSize();
        }
        void bind(double wjNew)
        {
            double &wj = (j == D ? b : w[j]);
            for(int i = 0; i < data.getSize(); ++i) sums[i] +=
                j == D ? (wjNew - b) : (wjNew - w[j]) * data.getX(i, j);
            wj = wjNew;
        }
    };
public:
    BinaryLSVM(pair<int, double> const& p): w(p.first), l(p.second), b(0),
        learnedCount(0) {}
    template<typename DATA> BinaryLSVM(DATA const& data, double theL,
        int nGoal = 100000, int nEvals = 100000): l(theL), b(0),
        w(getD(data)), learnedCount(0)
    {//first SGD
        for(int j = 0; j < ceiling(nGoal, data.getSize()); ++j)
            for(int i = 0; i < data.getSize(); ++i)
                learn(data.getX(i), data.getY(i), data.getSize());
        //then coordinate descent
        GSL1Functor<DATA> f(data, b, w, l);
        unimodalCoordinateDescent(f, nEvals, pow(10, -6));
    }
    int getLearnedCount(){return learnedCount;}
    void learn(NUMERIC_X const& x, int label, int n = -1)
    {//online mode uses SGD only
        if(n == -1) n = learnedCount + 1;
        double rate = RMRate(learnedCount++), yl = y(label);
        for(int i = 0; i < w.getSize(); ++i)
            w[i] -= rate * (w[i] > 0 ? 1 : -1) * l/n;
        if(yl * f(x) < 1)
        {
            w -= x * (-yl * rate);
            b += rate * yl;
        }
    }
    int predict(NUMERIC_X const& x)const{return f(x) >= 0;}
    template<typename MODEL, typename DATA>
    static double findL(DATA const& data)
    {//used for regression as well
        int lLow = -15, lHigh = 5;
        Vector<double> regs;
        for(double j = lHigh; j > lLow; j -= 2) regs.append(pow(2, j));
        return valMinFunc(regs.getArray(), regs.getSize(),
            SCVRiskFunctor<MODEL, double, DATA>(data));
    }
};

struct NoParamsLSVM
{
    typedef MulticlassLearner<BinaryLSVM, double> MODEL;
    MODEL model;
    template<typename DATA> NoParamsLSVM(DATA const& data): model(data,
        BinaryLSVM::findL<MODEL, DATA>(data)) {}
    int predict(NUMERIC_X const& x)const{return model.predict(x);}
};
typedef ScaledLearner<NoParamsLearner<NoParamsLSVM, int>, int> SLSVM;

class SRaceLSVM
{
    ScalerMQ s;
    typedef pair<int, double> P;
    RaceLearner<OnlineMulticlassLearner<BinaryLSVM, P>, P> model;
    static Vector<P> makeParams(int D)
    {
        Vector<P> result;
        int lLow = -15, lHigh = 5;
        for(int j = lHigh; j > lLow; j -= 2)
        {
            double l = pow(2, j);
            result.append(P(D, l));
        }
        return result;
    }
public:
    template<typename DATA> SRaceLSVM(DATA const& data):
        model(makeParams(getD(data))), s(getD(data))
    {
        for(int j = 0; j < 1000000; ++j)
        {
            int i = GlobalRNG().mod(data.getSize());
            learn(data.getX(i), data.getY(i));
        }
    }
    SRaceLSVM(int D): model(makeParams(D)), s(D){}
    void learn(NUMERIC_X const& x, int label)
    {
        s.addSample(x);
        model.learn(s.scale(x), label);
    }
    int predict(NUMERIC_X const& x)const{return model.predict(s.scale(x));}
};

struct GaussianKernel
{
    double a;
    GaussianKernel(double theA): a(theA) {}
    double operator()(NUMERIC_X const& x, NUMERIC_X const& y)const
    {//beware - if x - y is too large, result = 0
        NUMERIC_X temp = x - y;
        return exp(-a * dotProduct(temp, temp));
    }
};
template<typename KERNEL = GaussianKernel, typename X = NUMERIC_X> struct SVM
{
    Vector<X> supportVectors;
    Vector<double> supportCoefficients;
    double bias;
    KERNEL K;
    template<typename DATA> double evalK(LinearProbingHashTable<long long,
        double>& cache, long long i, long long j, DATA const& data)
    {
        long long key = i * data.getSize() + j;
        double* result = cache.find(key);
        if(result) return *result;
        else
        {
            double value = K(data.getX(i), data.getX(j));
            cache.insert(key, value);
            return value;
        }
    }
    int makeY(bool label){return label * 2 - 1;}
    double lowDiff(bool label, double C, double d){return label ? d : d + C;}
    double highDiff(bool label, double C, double d){return label ? C - d: d;}
public:
    template<typename DATA> SVM(DATA const& data, pair<KERNEL, double> const&
        params, int maxRepeats = 10, int maxConst = 10000): K(params.first)
    {
        double C = params.second;
        assert(data.getSize() > 0 && C > 0);
        bias = makeY(data.getY(0));//just in case have 1 class only
        LinearProbingHashTable<long long, double> cache;
        int n = data.getSize(), maxIters = max(maxConst, n * maxRepeats);
        Vector<double> d(n, 0), g(n);
        for(int k = 0; k < n; ++k) g[k] = makeY(data.getY(k));
        while(maxIters--)
        {//select directions using max violating pair
            int i = -1, j = -1;//i can increase, j can decrease
            for(int k = 0; k < n; ++k)
            {//find max gi and min gj
                if(highDiff(data.getY(k), C, d[k]) > 0 && (i == -1 ||
                    g[k] > g[i])) i = k;
                if(lowDiff(data.getY(k), C, d[k]) > 0 && (j == -1 ||
                    g[k] < g[j])) j = k;
            }
            if(i == -1 || j == -1) break;
            bias = (g[i] + g[j])/2;//ave for stability
            //check optimality condition
            double optGap = g[i] - g[j];
            if(optGap < 0.001) break;
            //compute direction-based minimum and box bounds
            double denom = evalK(cache, i, i, data) -
                2 * evalK(cache, i, j, data) + evalK(cache, j, j, data),
                step = min(highDiff(data.getY(i), C, d[i]),
                lowDiff(data.getY(j), C, d[j]));
            //shorten step to box bounds if needed, check for numerical
            //error in kernel calculation or duplicate data, if error
            //move points to box bounds
            if(denom > 0) step = min(step, optGap/denom);
            //update support vector coefficients and gradient
            d[i] += step;
            d[j] -= step;
            for(int k = 0; k < n; ++k) g[k] += step *
                (evalK(cache, j, k, data) - evalK(cache, i, k, data));
        }//determine support vectors
        for(int k = 0; k < n; ++k) if(abs(d[k]) > defaultPrecEps)
            {
                supportCoefficients.append(d[k]);
                supportVectors.append(data.getX(k));
            }
    }
    int predict(X const& x)const
    {
        double sum = bias;
        for(int i = 0; i < supportVectors.getSize(); ++i)
            sum += supportCoefficients[i] * K(supportVectors[i], x);
        return sum >= 0;
    }
};

template<typename KERNEL = GaussianKernel, typename X = NUMERIC_X>
class MulticlassSVM
{//need buffer for speed
    typedef pair<KERNEL, double> P;
    MulticlassLearner<BufferLearner<SVM<KERNEL, X>, InMemoryData<X, int>, P>,
        P> mcl;
public:
    template<typename DATA> MulticlassSVM(DATA const& data,
        pair<KERNEL, double> const& params): mcl(data, params) {}
    int predict(X const& x)const{return mcl.predict(x);}
};

struct NoParamsSVM
{
    MulticlassSVM<> model;
    struct CVSVMFunctor
    {
        typedef Vector<double> PARAMS;
        MulticlassSVM<> model;
        template<typename DATA> CVSVMFunctor(DATA const& data,
            PARAMS const& p):
            model(data, make_pair(GaussianKernel(p[0]), p[1])) {}
        int predict(NUMERIC_X const& x)const{return model.predict(x);}
    };
    template<typename DATA> static pair<GaussianKernel, double>
        gaussianMultiClassSVM(DATA const& data, int CLow = -5,
        int CHigh = 15, int yLow = -15, int yHi = 3)
    {
        Vector<Vector<double> > sets(2);
        for(int i = yLow; i <= yHi; i += 2) sets[0].append(pow(2, i));
        for(int i = CLow; i <= CHigh; i += 2) sets[1].append(pow(2, i));
        Vector<double> best = compassDiscreteMinimize(sets,
            SCVRiskFunctor<CVSVMFunctor, Vector<double>, DATA>(data),
            10);
        //Vector<double> best = gridMinimize(sets,
        //    SCVRiskFunctor<CVSVMFunctor, Vector<double> >(data));
        return make_pair(GaussianKernel(best[0]), best[1]);
    }
    template<typename DATA> NoParamsSVM(DATA const& data): model(data,
        gaussianMultiClassSVM(data)) {}
    int predict(NUMERIC_X const& x)const{return model.predict(x);}
};
typedef ScaledLearner<NoParamsLearner<NoParamsSVM, int>, int> SSVM;

template<typename LEARNER = RandomForest> class CostLearner
{
    Matrix<double> cost;
    LEARNER model;
public:
    template<typename DATA> CostLearner(DATA const& data,
        Matrix<double>const& costMatrix): model(data), cost(costMatrix) {}
    int predict(NUMERIC_X const& x)const
        {return costClassify(model.classifyProbs(x), cost);}
};

void scaleCostMatrix(Matrix<double>& cost)
{
    double maxCost = 0;
    for(int r = 0; r < cost.getRows(); ++r)
        for(int c = 0; c < cost.getRows(); ++c)
            maxCost = max(maxCost, cost(r, c));
    cost *= 1/maxCost;
}

template<typename LEARNER = NoParamsLearner<DecisionTree, int>,
    typename PARAMS = EMPTY, typename X = NUMERIC_X> class RMBoost
{
    Vector<LEARNER> classifiers;
    int nClasses;
    struct BinomialLoss
    {
        Vector<Vector<double> > F;
        BinomialLoss(int n, int nClasses): F(n, Vector<double>(nClasses, 0))
            {}
        int findBestFalse(int i, int label)
        {
            double temp = F[i][label];
            F[i][label] = -numeric_limits<double>::infinity();
            double result = argMax(F[i].getArray(), F[i].getSize());
            F[i][label] = temp;
            return result;
        }
        double getNegGrad(int i, int label, Matrix<double>const& costMatrix)
        {
            int bestFalseLabel = findBestFalse(i, label);
            double margin = F[i][label] - F[i][bestFalseLabel];
            return costMatrix(label, bestFalseLabel)/(exp(margin) + 1);
        }
    };
public:
    template<typename DATA> RMBoost(DATA const& data, Matrix<double>
        costMatrix, PARAMS const& p = PARAMS(),
        int nClassifiers = 100): nClasses(findNClasses(data))
    {//initial weights are based on ave cost
        int n = data.getSize();
        assert(n > 0 && nClassifiers > 0);
        BinomialLoss l(n, nClasses);
        Vector<double> dataWeights(n), classWeights(nClasses);
        for(int i = 0; i < nClasses; ++i)
            for(int j = 0; j < nClasses; ++j)
                classWeights[i] += costMatrix(i, j);
        for(int i = 0; i < n; ++i)
            dataWeights[i] = classWeights[data.getY(i)];
        for(int i = 0; i < nClassifiers; ++i)
        {
            normalizeProbs(dataWeights);
            AliasMethod sampler(dataWeights);
            PermutedData<DATA> resample(data);
            for(int j = 0; j < n; ++j) resample.addIndex(sampler.next());
            classifiers.append(LEARNER(resample, p));
            for(int j = 0; j < n; ++j)
            {
                l.F[j][classifiers.lastItem().predict(data.getX(j))] +=
                    RMRate(i);
                dataWeights[j] = l.getNegGrad(j, data.getY(j), costMatrix);
            }
        }
    }
    int predict(X const& x)const
    {
        Vector<double> counts(nClasses, 0);
        for(int i = 0; i < classifiers.getSize(); ++i)
            counts[classifiers[i].predict(x)] += RMRate(i);
        return argMax(counts.getArray(), counts.getSize());
    }
};

class BoostedCostSVM
{
    RMBoost<MulticlassSVM<>, pair<GaussianKernel, double> > model;
public:
    template<typename DATA> BoostedCostSVM(DATA const& data,
        Matrix<double> const& cost = Matrix<double>(1, 1)):
        model(data, cost, NoParamsSVM::gaussianMultiClassSVM(data), 15) {}
    int predict(NUMERIC_X const& x)const{return model.predict(x);}
};
typedef ScaledLearner<BoostedCostSVM, int, Matrix<double> > SBoostedCostSVM;

template<typename LEARNER, typename PARAMS = EMPTY,
    typename X = NUMERIC_X> class AveCostLearner
{
    LEARNER model;
    template<typename DATA> static Vector<double> findWeights(
        DATA const& data, Matrix<double> const& costMatrix)
    {//init with average weights
        int k = costMatrix.getRows(), n = data.getSize();
        assert(k > 1 && k == findNClasses(data));
        Vector<double> classWeights(k), result(n);
        for(int i = 0; i < k; ++i)
            for(int j = 0; j < k; ++j)
                classWeights[i] += costMatrix(i, j);
        for(int i = 0; i < n; ++i) result[i] = classWeights[data.getY(i)];
        normalizeProbs(result);
        return result;
    }
public:
    template<typename DATA> AveCostLearner(DATA const& data,
        Matrix<double> const& costMatrix, PARAMS const& p = PARAMS()):
        model(data, findWeights(data, costMatrix), p) {}
    int predict(X const& x)const{return model.predict(x);}
};

class AveCostSVM
{
    typedef pair<GaussianKernel, double> P;
    AveCostLearner<WeightedBaggedLearner<MulticlassSVM<>, P>, P> model;
public:
    template<typename DATA> AveCostSVM(DATA const& data,
        Matrix<double>const& cost = Matrix<double>(1, 1)):
        model(data, cost, NoParamsSVM::gaussianMultiClassSVM(data)) {}
    int predict(NUMERIC_X const& x)const{return model.predict(x);}
};
typedef ScaledLearner<AveCostSVM, int, Matrix<double> > SAveCostSVM;

class ImbalanceSVM
{
    WeightedBaggedLearner<MulticlassSVM<>,
        pair<GaussianKernel, double> > model;
public:
    template<typename DATA> ImbalanceSVM(DATA const& data): model(data,
        findImbalanceWeights(data), NoParamsSVM::gaussianMultiClassSVM(data))
        {}
    int predict(NUMERIC_X const& x)const{return model.predict(x);}
};
typedef ScaledLearner<NoParamsLearner<ImbalanceSVM, int>, int> SImbSVM;

/*
template<typename DISTANCE = EuclideanDistance<NUMERIC_X>::Distance>
struct MeanNN
{
    Vector<NUMERIC_X> means;
    DISTANCE d;
public:
    template <typename DATA> MeanNN(DATA const& data,
        DISTANCE const& theD = DISTANCE()): d(theD)
    {
        int D = getD(data), nClasses = findNClasses(data);
        NUMERIC_X zero(D);
        Vector<int> counts(nClasses);
        means = Vector<NUMERIC_X>(nClasses, zero);
        for(int i = 0; i < data.getSize(); ++i)
        {
            int y = data.getY(i);
            means[y] += data.getX(i);
            ++counts[y];
        }
        for(int i = 0; i < nClasses; ++i)
            if(counts[i] > 0) means[i] *= 1.0/counts[i];
    }
public:
    int predict(NUMERIC_X const& x)const
    {
        int best = -1;
        double bestD;
        for(int i = 0; i < means.getSize(); ++i)
        {
            double dist = d(means[i], x);
            if(best == -1 || dist < bestD)
            {
                best = i;
                bestD = dist;
            }
        }
        return best;
    }
};*/

template<typename SUBSET_LEARNER = RandomForest> struct SmartFSLearner
{
    typedef FeatureSubsetLearner<SUBSET_LEARNER> MODEL;
    MODEL model;
public:
    template<typename DATA> SmartFSLearner(DATA const& data, int limit = 20):
        model(data, selectFeaturesSmart(SCVRiskFunctor<MODEL, Bitset<>,DATA>(
        data), getD(data), limit)) {}
    int predict(NUMERIC_X const& x)const{return model.predict(x);}
};

class BinaryNN
{
    Vector<NeuralNetwork> nns;
    void setupStructure(int D, int nHidden)
    {
        double a = sqrt(3.0/D);
        for(int l = 0; l < nns.getSize(); ++l)
        {
            NeuralNetwork& nn = nns[l];
            nn.addLayer(nHidden);
            for(int j = 0; j < nHidden; ++j)
                for(int k = -1; k < D; ++k) nn.addConnection(0, j, k,
                    k == -1 ? 0 : GlobalRNG().uniform(-a, a));
            nn.addLayer(1);
            for(int k = -1; k < nHidden; ++k) nn.addConnection(1, 0, k, 0);
        }
    }
public:
    BinaryNN(int D, int nHidden = 5, int nNns = 5):
        nns(nNns, NeuralNetwork(D)){setupStructure(D, nHidden);}
    template<typename DATA> BinaryNN(DATA const& data, int nHidden = 5, int
        nGoal = 100000, int nNns = 5): nns(nNns, NeuralNetwork(getD(data)))
    {
        int D = getD(data), nRepeats = ceiling(nGoal, data.getSize());
        setupStructure(D, nHidden);
        for(int j = 0; j < nRepeats; ++j)
            for(int i = 0; i < data.getSize(); ++i)
                learn(data.getX(i), data.getY(i));
    }
    void learn(NUMERIC_X const& x, int label)
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
    int predict(NUMERIC_X const& x)const{return evaluate(x) > 0.5;}
};

class MulticlassNN
{
    MulticlassLearner<NoParamsLearner<BinaryNN, int>, EMPTY> model;
public:
    template<typename DATA> MulticlassNN(DATA const& data): model(data) {}
    int predict(NUMERIC_X const& x)const {return model.classifyByProbs(x);}
};
typedef ScaledLearner<NoParamsLearner<MulticlassNN, int>, int, EMPTY,
    ScalerMQ> SNN;

class SOnlineNN
{
    ScalerMQ s;
    OnlineMulticlassLearner<BinaryNN, int> model;
public:
    template<typename DATA> SOnlineNN(DATA const& data): model(getD(data)),
        s(getD(data))
    {
        for(int j = 0; j < 1000000; ++j)
        {
            int i = GlobalRNG().mod(data.getSize());
            learn(data.getX(i), data.getY(i));
        }
    }
    SOnlineNN(int D): model(D), s(D) {}
    void learn(NUMERIC_X const& x, int label)
    {
        s.addSample(x);
        model.learn(s.scale(x), label);
    }
    int predict(NUMERIC_X const& x)const{return model.predict(s.scale(x));}
};

/*
class DeepNN
{
    int D, nClasses;
    DeepNeuralNetwork nn;
public:
    void learnUnsupI(NUMERIC_X const& x, int i, int n){nn.learnUnsupI(x, i, n);}
    template<typename MODEL> static int findNHidden(
        typename DATA<>::LABELED_DATA const& data)
    {
        int nLow = 1, nHigh = 6;
        Vector<int> sizes;
        for(int j = nLow; j <= nHigh; ++j)
        {
            int size = pow(2, j) + 1;
            sizes.append(size);
        }
        return valMinFunc(sizes.getArray(), sizes.getSize(),
            SCVRiskFunctor<MODEL, int>(data));
    }
    DeepNN(DATA<>::LABELED_DATA const& data, int nHidden,
        int nGoal = 1000000, double l = 0): D(data[0].first.getSize()),
        nClasses(findNClasses<NUMERIC_X>(data)), nn(D, nClasses, nHidden, l)
    {
        int nTrains = max(nGoal, data.getSize());
        for(int i = 0; i < nn.nHiddenLayers(); ++i)
        {
            for(int k = 0; k < nTrains; ++k)
            {
                int k = GlobalRNG().mod( data.getSize());
                learnUnsupI(data[k].first, i, data.getSize());
            }
        }
        while(nTrains--)
        {
            int i = GlobalRNG().mod( data.getSize());
            learn(data[i].first, data[i].second, data.getSize());
        }
    }
    void learn(NUMERIC_X const& x, int label, int n)
    {
        assert(label >= 0 && label < nClasses);
        Vector<double> result(nClasses);
        result[label] = 1;
        nn.learn(x, result, n);
    }
    int predict(NUMERIC_X const& x)const
    {
        Vector<double> result = nn.evaluate(x);
        return argMax(result.getArray(), result.getSize());
    }
};
class NoParamDeepNN
{
    DeepNN model;
public:
    NoParamDeepNN(DATA<>::LABELED_DATA const& data): model(data, DeepNN::findNHidden<DeepNN>(data)){}
    int predict(NUMERIC_X const& x)const {return model.predict(x);}
};*/

class SimpleBestCombiner
{
    BestCombiner<int> c;
public:
    template<typename DATA> SimpleBestCombiner(DATA const& data)
    {
        c.addNoParamsClassifier<RandomForest>(data, SCVRiskFunctor<
            NoParamsLearner<RandomForest, int>, EMPTY, DATA>(data));
        c.addNoParamsClassifier<SSVM>(data, SCVRiskFunctor<
            NoParamsLearner<SSVM, int>, EMPTY, DATA>(data));
    }
    int predict(NUMERIC_X const& x)const{return c.predict(x);}
};

}//end namespace
#endif

