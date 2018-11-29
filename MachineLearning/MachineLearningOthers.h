#ifndef MACHINELEARNINGOTHER_H
#define MACHINELEARNINGOTHER_H
#include "LearningCommon.h"
#include "../Utils/Utils.h"
#include "../MiscAlgs/Misc.h"
#include "../HashTable/LinearProbingHashTable.h"
#include "../ComputationalGeometry/KDTree.h"
#include "../ComputationalGeometry/Point.h"
#include "../NumericalMethods/Matrix.h"
#include "../NumericalMethods/SparseMatrix.h"
#include "../RandomTreap/LCPTreap.h"
#include "../RandomNumberGeneration/Statistics.h"
#include "../Graphs/Graph.h"
#include <cmath>
namespace igmdk{

template<typename DATA, typename DISTANCE, typename REPS>
double clusterSimplifiedSilhouette(DATA const& data,
    Vector<int> const& assignments, REPS const& r, DISTANCE const & d)
{
    int n = assignments.getSize();
    assert(n > 0);
    int k = valMax(assignments.getArray(), n) + 1;
    double sum = 0;
    for(int i = 0; i < n; ++i)
    {
        int c = assignments[i];
        double ai = d(data.getX(i), r[c]),
            bi = numeric_limits<double>::infinity();
        for(int j = 0; j < k; ++j) if(j != c)
            bi = min(bi, d(data.getX(i), r[j]));
        sum += (bi - ai)/max(bi, ai);
    }
    return sum/n;
}
template<typename DATA> double clusterSimplifiedSilhouetteL2(DATA const& data,
    Vector<int> const& assignments)
{
    int k = valMax(assignments.getArray(), assignments.getSize()) + 1;
    return clusterSimplifiedSilhouette(data, assignments, findCentroids(data,
        k, assignments), EuclideanDistance<NUMERIC_X>::Distance());
}

template<typename DATA, typename DISTANCE> double clusterSilhouette(
    DATA const& data, Vector<int> const& assignments, DISTANCE const& d)
{
    int n = assignments.getSize();
    assert(n > 0);
    int k = valMax(assignments.getArray(), n) + 1;
    double sum = 0;
    for(int i = 0; i < n; ++i)
    {
        int c = assignments[i];
        Vector<double> ds(k);
        Vector<int> sizes(k);
        for(int j = 0; j < n; ++j) if(i != j)
            {
                int c2 = assignments[j];
                ++sizes[c2];
                ds[c2] += d(data.getX(i), data.getX(j));
            }
        for(int j = 0; j < k; ++j) if(sizes[j]) ds[j] /= sizes[j];
        double ai = ds[c], bi = numeric_limits<double>::infinity();
        for(int j = 0; j < k; ++j) if(j != c) bi = min(bi, ds[j]);
        sum += (bi - ai)/max(bi, ai);
    }
    return sum/n;
}

struct ClusterResult
{
    Vector<int> assignments;
    double comparableInternalIndex;
    ClusterResult(Vector<int> const& theAssignments, double theCIP =
        numeric_limits<double>::infinity()): assignments(theAssignments),
        comparableInternalIndex(theCIP){}
};
template<typename CLUSTERER, typename DATA, typename PARAMS> ClusterResult
    findClustersAndK(DATA const& data, CLUSTERER const& c, PARAMS const& p,
    int maxK = -1)
{
    if(maxK == -1) maxK = sqrt(data.getSize());
    Vector<int> dummy;
    ClusterResult best(dummy);
    for(int k = 2; k <= maxK; ++k)
    {
        ClusterResult result = c(data, k, p);
        if(isfinite(result.comparableInternalIndex) &&
            result.comparableInternalIndex < best.comparableInternalIndex)
            best = result;
        else break;
    }
    return best;
}
template<typename CLUSTERER, typename PARAMS = EMPTY> struct FindKClusterer
{
    CLUSTERER c;
    PARAMS p;
    FindKClusterer(PARAMS const& theP = PARAMS()): p(theP){}
    template<typename DATA> ClusterResult operator()(DATA const& data, int k)
        const{return c(data, k, p);}
    template<typename DATA> ClusterResult operator()(DATA const& data)const
        {return findClustersAndK(data, c, p);}
};

template<typename CLUSTERER> struct NoParamsClusterer
{
    CLUSTERER c;
    template<typename DATA> ClusterResult operator()(DATA const& data, int k,
        EMPTY const& p)const{return c(data, k);}
    template<typename DATA> ClusterResult operator()(DATA const& data,
        EMPTY const& p)const{return c(data);}
};

template<typename DATA> Vector<NUMERIC_X> findCentroids(DATA const& data,
    int k, Vector<int> const& assignments)
{
    Vector<int> counts(k);
    Vector<NUMERIC_X> centroids(k, data.getX(0) * 0);
    for(int i = 0; i < data.getSize(); ++i)
    {
        ++counts[assignments[i]];
        centroids[assignments[i]] += data.getX(i);
    }
    for(int i = 0; i < k; ++i) centroids[i] *= 1.0/counts[i];
    return centroids;
}

template<typename DATA, typename DISTANCE> Vector<int>
    findKMeansPPCentoids(DATA const& data, int k, DISTANCE const& d,
    bool isMetric = false)
{//approximation algorithm to initialize centroids
    int n = data.getSize();
    assert(n > 0 && k <= n);
    Vector<double> closestDistances(n, numeric_limits<double>::infinity());
    Vector<int> centroids(1, GlobalRNG().mod(n));
    for(int i = 1; i < k; ++i)
    {//recompute closest center distances
        for(int j = 0; j < n; ++j)
            closestDistances[j] = min(closestDistances[j],
                d(data.getX(j), data.getX(centroids.lastItem())));
        //sample next center in proportion to squared closest distance
        Vector<double> probs(n);
        for(int j = 0; j < n; ++j)
        {
            probs[j] = closestDistances[j];
            if(!isMetric) probs[j] *= closestDistances[j];
        }
        normalizeProbs(probs);
        AliasMethod a(probs);
        centroids.append(a.next());
    }
    return centroids;
}
template<typename DATA> Vector<typename DATA::X_TYPE> assemblePrototypes(
    DATA const& data, Vector<int> const& medoids)
{//helper to get cluster centers from data and indices
    Vector<typename DATA::X_TYPE> result(medoids.getSize());
    for(int i = 0; i < medoids.getSize(); ++i)
        result[i] = data.getX(medoids[i]);
    return result;
}

struct KMeans
{
    typedef EuclideanDistance<NUMERIC_X>::Distance EUC_D;
    template<typename DATA> static bool findAssigments(DATA const& data,
        Vector<NUMERIC_X> const& centroids, Vector<int>& assignments)
    {//assign all examples to their nearest centroids
        VpTree<NUMERIC_X, int, EUC_D> t;
        bool converged = true;
        for(int i = 0; i < centroids.getSize(); ++i) t.insert(centroids[i], i);
        //assign each point to the closest centroid
        for(int i = 0; i < data.getSize(); ++i)
        {
            int best = t.nearestNeighbor(data.getX(i))->value;
            if(best != assignments[i])
            {//done if no assignment changed
                converged = false;
                assignments[i] = best;
            }
        }
        return converged;
    }
    template<typename DATA> ClusterResult operator()(DATA const& data,
        int k, int maxIterations = 1000)const
    {
        assert(k > 0 && k <= data.getSize() && data.getSize() > 0);
        Vector<int> assignments(data.getSize());
        findAssigments(data, assemblePrototypes(data, findKMeansPPCentoids(
            data, k, EUC_D())), assignments);
        for(int m = 0; m < maxIterations; ++m) if(findAssigments(data,
            findCentroids(data, k, assignments), assignments)) break;
        return ClusterResult(assignments, -clusterSimplifiedSilhouetteL2(data,
            assignments));
    }
};
typedef FindKClusterer<NoParamsClusterer<KMeans> > KMeansGeneral;

template<typename DATA> double kMeansSimpSil(DATA const& data,
    Vector<int> const& assignments)
{
    int n = assignments.getSize();
    assert(n > 0);
    int k = valMax(assignments.getArray(), n) + 1;
    Vector<NUMERIC_X> centroids = findCentroids(data, k, assignments);
    typename EuclideanDistance<NUMERIC_X>::DistanceIncremental d;
    double sum = 0;
    for(int i = 0; i < n; ++i)
        sum += d(data.getX(i), centroids[assignments[i]]);
    return sum;
}
struct RepeatedKMeans
{
    template<typename DATA> ClusterResult operator()(DATA const& data,
        int k, int maxIterations = 1000, int nRep = 10)const
    {
        KMeans km;
        ClusterResult best = km(data, k, maxIterations);
        double ss = kMeansSimpSil(data, best.assignments);
        while(--nRep)
        {
            ClusterResult result = km(data, k, maxIterations);
            double ssNew = kMeansSimpSil(data, result.assignments);
            if(ssNew < ss)
            {
                ss = ssNew;
                best = result;
            }
        }
        return best;
    }
};
typedef FindKClusterer<NoParamsClusterer<RepeatedKMeans> > RKMeansGeneral;

template<typename DISTANCE = EuclideanDistance<NUMERIC_X>::Distance>
struct KMedoids
{
    static bool isIMedoid(int i, Vector<int> const& medoids)
    {
        for(int j = 0; j < medoids.getSize(); ++j)
            if(medoids[j] == i) return true;
        return false;
    }
    template<typename DATA> ClusterResult operator()(DATA const& data, int k,
        int maxRounds = 1000)const
        {return findClusters(data, k, maxRounds).first;}
    template<typename DATA> static pair<ClusterResult, Vector<int> >
        findClusters(DATA const& data, int k, int maxRounds = 1000)
    {//initialize current medoids
        int n = data.getSize();
        assert(k > 0 && k <= n && n > 0);
        DISTANCE d;
        Vector<int> perm(n), medoids(findKMeansPPCentoids(data, k, d, true)),
            assignments(n);
        Vector<Vector<double> > dCache(n, Vector<double>(k));
        for(int i = 0; i < n; ++i)
        {//compute current assignments and cache the distances
            for(int j = 0; j < k; ++j)
                dCache[i][j] = d(data.getX(i), data.getX(medoids[j]));
            int best = argMin(dCache[i].getArray(), k);
            assignments[i] = best;
        }
        for(int i = 0; i < n; ++i) perm[i] = i;//initialize the permutation
        bool converged = false;
        while(!converged)
        {
            converged = true;
            GlobalRNG().randomPermutation(perm.getArray(), n);
            for(int i = 0; i < n && maxRounds > 0; ++i)
            {
                if(isIMedoid(perm[i], medoids)) continue;
                Vector<double> tempDs(n);
                for(int l = 0; l < n; ++l)
                    tempDs[l] = d(data.getX(perm[i]), data.getX(l));
                int bestJ = -1;
                double bestDiff;
                for(int j = 0; j < k; ++j)
                {
                    double DSumDiff = 0;
                    for(int l = 0; l < n; ++l)
                    {
                        double dOld = dCache[l][assignments[l]];
                        if(assignments[l] == j)
                        {
                            dCache[l][j] = tempDs[l];
                            DSumDiff += valMin(dCache[l].getArray(), k) - dOld;
                            dCache[l][j] = dOld;
                        }
                        else if(tempDs[l] < dOld) DSumDiff += tempDs[l] - dOld;
                    }
                    if(bestJ == -1 || DSumDiff < bestDiff)
                    {
                        bestDiff = DSumDiff;
                        bestJ = j;
                    }
                }
                if(bestDiff < 0)
                {
                    converged = false;
                    for(int l = 0; l < n; ++l)
                    {
                        dCache[l][bestJ] = tempDs[l];
                        if(assignments[l] == bestJ) assignments[l] =
                            argMin(dCache[l].getArray(), k);
                        else if(tempDs[l] < dCache[l][assignments[l]])
                            assignments[l] = bestJ;
                    }
                    medoids[bestJ] = perm[i];
                }
                --maxRounds;
            }
        }
        return make_pair(ClusterResult(assignments,
            -clusterSimplifiedSilhouette(data, assignments,
            assemblePrototypes(data, medoids), d)), medoids);
    }
};
template<typename DISTANCE = EuclideanDistance<NUMERIC_X>::Distance>
using KMedGeneral = FindKClusterer<NoParamsClusterer<KMedoids<DISTANCE> > >;

template<typename DISTANCE = EuclideanDistance<NUMERIC_X>::Distance>
struct RepeatedKMedoids
{
    template<typename DATA> static double kMedS(DATA const& data,
        Vector<int> const& assignments, Vector<int> const& medoids)
    {
        int n = assignments.getSize();
        assert(n > 0);
        int k = valMax(assignments.getArray(), n) + 1;
        Vector<typename DATA::X_TYPE> m = assemblePrototypes(data, medoids);
        DISTANCE d;
        double sum = 0;
        for(int i = 0; i < n; ++i) sum += d(data.getX(i), m[assignments[i]]);
        return sum;
    }
    template<typename DATA> ClusterResult operator()(DATA const& data,
        int k, int maxRounds = 1000, int nRep = 10)const
    {
        KMedoids<DISTANCE> km;
        pair<ClusterResult, Vector<int> > best =
            KMedoids<DISTANCE>::findClusters(data, k, maxRounds);
        double s = kMedS(data, best.first.assignments, best.second);
        while(--nRep)
        {
            pair<ClusterResult, Vector<int> > result =
                KMedoids<DISTANCE>::findClusters(data, k, maxRounds);
            double sNew = kMedS(data, result.first.assignments, result.second);
            if(sNew < s)
            {
                s = sNew;
                best.first = result.first;
            }
        }
        return best.first;
    }
};
template<typename DISTANCE = EuclideanDistance<NUMERIC_X>::Distance>
using RKMedGeneral =
    FindKClusterer<NoParamsClusterer<RepeatedKMedoids<DISTANCE> > >;

template<typename DISTANCE = EuclideanDistance<NUMERIC_X>::Distance>
struct SpectralClusterer
{//eigenpairs and permutations for sorting
    typedef pair<pair<Vector<double>, Matrix<double> >, Vector<int> > EIGS;
    template<typename DATA> SparseMatrix<double> createLaplacian(
        DATA const& data)const
    {//setup kNN Laplacian
        int n = data.getSize(), nNeighbors = lgFloor(n)/2 + 1;//kNN default
        SparseMatrix<double> W(n, n);
        typedef VpTree<typename DATA::X_TYPE, int, DISTANCE> TREE;
        TREE tree;
        for(int i = 0; i < n; ++i) tree.insert(data.getX(i), i);
        for(int i = 0; i < n; ++i)
        {
            Vector<typename TREE::NodeType*> neighbors =
                tree.kNN(data.getX(i), nNeighbors + 1);
            for(int j = 0; j < neighbors.getSize(); ++j)
            {//first nn is usually self
                int l = neighbors[j]->value;
                if(l != i)
                {
                    W.set(i, l, 1);
                    W.set(l, i, 1);
                }
            }
        }
        return W;
    }
    EIGS findLaplacianEigs(SparseMatrix<double> const& W)const
    {//normalize
        int n = W.getRows();
        SparseMatrix<double> Dm05(n, n);
        for(int i = 0; i < n; ++i)
        {
            double di = 0;
            for(int j = 0; j < n; ++j) di += W(i, j);
            Dm05.set(i, i, 1/sqrt(di));
        }//find eigs
        EIGS eigs(QREigenSymmetric(toDense<double>(SparseMatrix<double>::
            identity(n) - Dm05 * W * Dm05)), Vector<int>(n));
        //sort the permutation
        for(int i = 0; i < n; ++i) eigs.second[i] = i;
        quickSort(eigs.second.getArray(), 0, n - 1,
            IndexComparator<double>(eigs.first.first.getArray()));
        /*//Lanczos experiment
        EIGS eigs2(LanczosEigenSymmetric(SparseMatrix<double>::
            identity(n) - Dm05 * W * Dm05), Vector<int>(n));
        for(int i = 0; i < n; ++i) eigs2.second[i] = i;
        quickSort(eigs2.second.getArray(), 0, n - 1,
            IndexComparator<double>(eigs2.first.first.getArray()));
        int m = 10;
        EIGS eigs3(LanczosEigenSymmetric(SparseMatrix<double>::
            identity(n) - Dm05 * W * Dm05, m), Vector<int>(m));
        for(int i = 0; i < m; ++i)
        {
            DEBUG(eigs.first.first[eigs.second[i]]);
            DEBUG(eigs2.first.first[eigs2.second[i]]);
            DEBUG(eigs3.first.first[i]);
        }*/
        return eigs;
    }
    template<typename DATA> ClusterResult operator()(DATA const& data, int k,
        EIGS const& eigs)const//to be called by k search
    {//make new features
        int n = data.getSize();
        assert(k > 0 && k < n);
        InMemoryData<NUMERIC_X, int> data2;
        for(int i = 0; i < n; ++i)
        {//eigenvectors are rows
            Vector<double> x(k);
            for(int j = 0; j < k; ++j)
                x[j] = eigs.first.second(eigs.second[j], i);
            double xNorm = norm(x);//normalize
            if(xNorm > 0) x *= 1/xNorm;
            data2.addZ(x, 0);
        }//cluster
        RepeatedKMeans km;
        ClusterResult result = km(data2, k);
        result.comparableInternalIndex =
            -clusterSilhouette(data, result.assignments, DISTANCE());
        return result;
    }
    template<typename DATA> ClusterResult operator()(DATA const& data, int k)
        const//if know k
        {return operator()(data, k, findLaplacianEigs(createLaplacian(data)));}
    template<typename DATA> ClusterResult operator()(DATA const& data)const
    {//if don't know k
        return findClustersAndK(data, *this,
            findLaplacianEigs(createLaplacian(data)));
    }
};
template<typename DISTANCE = EuclideanDistance<NUMERIC_X>::Distance>
struct SpectralSmart
{
    RKMedGeneral<DISTANCE> km;
    SpectralClusterer<DISTANCE> s;
    template<typename DATA> bool useKMed(DATA const& data)const
        {DEBUG(data.getSize()); return data.getSize() > 5000;}//for memory
    //feasibility + efficiency
    template<typename DATA> ClusterResult operator()(DATA const& data,
        int k)const
    {
        if(useKMed(data)) return km(data, k);
        else return s(data, k);
    }
    template<typename DATA> ClusterResult operator()(DATA const& data)const
    {
        if(useKMed(data)) return km(data);
        return s(data);
    }
};

struct EMClustering
{
    static double normalLL(Vector<double> x, Vector<double> const& m,
        Cholesky<double> const& l)
    {
        x -= m;
        return -(x.getSize() * log(2 * PI()) + l.logDet() +
            dotProduct(l.solve(x), x))/2;
    }
    static double llim(){return log(numeric_limits<double>::min())/2;}
    static pair<Vector<double>, double> findILL(NUMERIC_X const& x,
        Vector<double> const& w, Vector<Vector<double> > const& m,
        Vector<Cholesky<double> > const &ls)
    {
        int k = w.getSize();
        Vector<double> temp(k);
        for(int j = 0; j < k; ++j)
            temp[j] = w[j] > 0 ? log(w[j]) + normalLL(x, m[j], ls[j]) : 0;
        double b = argMax(temp.getArray(), k);
        for(int j = 0; j < k; ++j) temp[j] -= b;
        return make_pair(temp, b);
    }
    template<typename DATA> static double findLL(DATA const& data,
        Vector<double> const& w, Vector<Vector<double> > const& m,
        Vector<Cholesky<double> > const &ls)
    {
        double ll = 0;
        for(int i = 0; i < data.getSize(); ++i)
        {
            pair<Vector<double>, double> temp =
                findILL(data.getX(i), w, m, ls);
            double kSum = temp.second;
            for(int j = 0; j < w.getSize(); ++j) kSum += exp(temp.first[j]);
            ll += log(kSum);
        }
        return ll;
    }
    template<typename DATA> ClusterResult operator()(DATA const& data, int k,
        int maxIterations = 1000)const
    {
        int n = data.getSize(), D = getD(data);
        assert(k > 0 && k <= n && n > 0);
        //initial values
        Vector<int> assignments =
            RepeatedKMeans()(data, k, maxIterations).assignments;
        Vector<Vector<double> > m = findCentroids(data, k, assignments),
            g(n, Vector<double>(k));
        Vector<double> w(k, 1.0/k);
        double pooledVar = 0;
        for(int i = 0; i < n; ++i)
        {
            Vector<double> diff = (data.getX(i) - m[assignments[i]]);
            pooledVar += dotProduct(diff, diff);
        }
        pooledVar /= (n - k) * D;
        Vector<Cholesky<double> > ls(k,
            Cholesky<double>(Matrix<double>::identity(D) * pooledVar));
        double ll = findLL(data, w, m, ls);
        while(maxIterations--)
        {//E step
            for(int i = 0; i < n; ++i)
            {
                pair<Vector<double>, double> temp =
                    findILL(data.getX(i), w, m, ls);
                for(int j = 0; j < k; ++j) g[i][j] = exp(temp.first[j]);
                normalizeProbs(g[i]);
            }
            //M step
            bool isNumericIssue = false;
            for(int j = 0; j < k; ++j)
            {//nj
                double nj = 0;
                for(int i = 0; i < n; ++i) nj += g[i][j];
                if(nj < 1){isNumericIssue = true; break;}
                //w
                w[j] = nj/n;
                //m
                for(int i = 0; i < n; ++i) m[j] += data.getX(i) * g[i][j];
                m[j] *= 1/nj;
                //average in pooled variance
                Matrix<double> covar = Matrix<double>::identity(D) *
                    pooledVar;
                for(int i = 0; i < n; ++i)
                {
                    Vector<double> xm = data.getX(i) - m[j];
                    covar += outerProduct(xm, xm) * g[i][j];
                }
                covar *= 1/(1 + nj);
                ls[j] = Cholesky<double>(covar);
                if(ls[j].failed){isNumericIssue = true; break;}
            }
            if(isNumericIssue) break;
            double newLL = findLL(data, w, m, ls);
            if(!isfinite(newLL) || !isELess(ll, newLL, 0.00001)) break;
            ll = newLL;
        }
        double BIC = -2 * ll + (k * (1 + D + D * (D + 1)/2) - 1) * log(n);
        for(int i = 0; i < n; ++i)
            assignments[i] = argMax(g[i].getArray(), k);
        return ClusterResult(assignments, BIC);
    }
};
typedef FindKClusterer<NoParamsClusterer<EMClustering> > EMBIC;
struct EMSmart
{
    KMeansGeneral km;
    EMBIC em;
    template<typename DATA> bool useKMeans(DATA const& data)const
        {return getD(data) > 100;}//for efficiency and numerics
    template<typename DATA> ClusterResult operator()(DATA const& data,
        int k)const
    {
        if(useKMeans(data)) return km(data, k);
        else return em(data, k);
    }
    template<typename DATA> ClusterResult operator()(DATA const& data)const
    {
        if(useKMeans(data)) return km(data);
        return em(data);
    }
};

template<typename DATA> Matrix<int> clusterContingencyMatrix(
    Vector<int> const& assignments, DATA const& data)
{//row is assignment, column is label
    int n = assignments.getSize();
    assert(n == data.getSize());
    int k = valMax(assignments.getArray(), n) + 1,
        nClasses = findNClasses(data);
    Matrix<int> counts(k, nClasses);
    for(int i = 0; i < n; ++i) ++counts(assignments[i], data.getY(i));
    return counts;
}

double clusterPurity(Matrix<int> const& counts)
{
    int sum = 0, total = 0;
    for(int i = 0; i < counts.rows; ++i)
    {
        int maxI = 0;
        for(int j = 0; j < counts.columns; ++j)
        {
            total += counts(i, j);
            maxI = max(maxI, counts(i, j));
        }
        sum += maxI;
    }
    return sum * 1.0/total;
}

double clusterClassificationAccuracy(Matrix<int> const& counts)
{
    int n = 0, sum = 0;
    for(int i = 0; i < counts.rows; ++i)
        for(int j = 0; j < counts.columns; ++j) n += counts(i, j);
    Vector<pair<pair<int, int>, double> > allowedMatches;
    for(int i = 0; i < counts.rows; ++i)
        for(int j = 0; j < counts.columns; ++j) allowedMatches.append(
            make_pair(make_pair(i, counts.rows + j), n - counts(i, j)));
    Vector<pair<int, int> > matches = assignmentProblem(counts.rows,
        counts.columns, allowedMatches);
    assert(matches.getSize() == min(counts.rows, counts.columns));
    for(int i = 0; i < matches.getSize(); ++i)
        sum += counts(matches[i].first, matches[i].second - counts.rows);
    return sum * 1.0/n;
}

double nChoose2(int n){return n * (n - 1)/2;}
double AdjustedRandIndex(Matrix<int> const& counts)
{
    int total = 0, SumRC2 = 0, SumR2 = 0, SumC2 = 0;
    for(int i = 0; i < counts.rows; ++i)
    {
        int sumI = 0;
        for(int j = 0; j < counts.columns; ++j)
        {
            int c = counts(i, j);
            total += c;
            sumI += c;
            SumRC2 += nChoose2(c);
        }
        SumR2 += nChoose2(sumI);
    }
    for(int j = 0; j < counts.columns; ++j)
    {
        int sumJ = 0;
        for(int i = 0; i < counts.rows; ++i) sumJ += counts(i, j);
        SumC2 += nChoose2(sumJ);
    }
    double EV = SumR2 * SumC2 * 1.0/nChoose2(total);
    return (SumRC2 - EV)/(0.5 * (SumR2 + SumC2) - EV);
}

Matrix<int> clusterOnlyContingencyMatrix(Vector<int> const& assignments1,
    Vector<int> const& assignments2)
{
    int n = assignments1.getSize();
    assert(n == assignments2.getSize());
    int k1 = valMax(assignments1.getArray(), n) + 1,
        k2 = valMax(assignments2.getArray(), n) + 1;
    Matrix<int> counts(k1, k2);
    for(int i = 0; i < n; ++i) ++counts(assignments1[i], assignments2[i]);
    return counts;
}
template<typename CLUSTERER, typename PARAMS, typename DATA> double
    findStability(CLUSTERER const &c, PARAMS const& p, DATA const& data,
    int k = -1, int B = 100)
{
    double sum = 0;
    int n = data.getSize();
    for(int j = 0; j < B; ++j)
    {//draw and cluster bootstraps
        Vector<int> assignments[2] = {Vector<int>(n, -1), Vector<int>(n, -1)};
        for(int l = 0; l < 2; ++l)
        {
            PermutedData<DATA> dataP(data);
            for(int i = 0; i < n; ++i) dataP.addIndex(GlobalRNG().mod(n));
            Vector<int> pAssignments = k == -1 ? c(dataP, p).assignments :
                c(dataP, k, p).assignments;
            for(int i = 0; i < n; ++i)
                assignments[l][dataP.permutation[i]] = pAssignments[i];
        }//compute and score intersection
        for(int i = n - 1; i >= 0; --i)
            if(assignments[0][i] == -1 || assignments[1][i] == -1)
                for(int l = 0; l < 2; ++l)
                {
                    assignments[l][i] = assignments[l].lastItem();
                    assignments[l].removeLast();
                }
        double temp = AdjustedRandIndex(clusterOnlyContingencyMatrix(
            assignments[0], assignments[1]));
        if(isfinite(temp)) sum += temp;
    }
    return sum/B;
}

template<typename CLASSIFIER, typename DATA> double findTestCAcc(
    DATA const& train, Vector<int> assignments, DATA const& test)
{
    assert(train.getSize() == assignments.getSize());
    RelabeledData<DATA> rd(train);
    rd.labels = assignments;
    CLASSIFIER c(rd);
    Vector<int> assignmentsTest(test.getSize());
    for(int i = 0; i < test.getSize(); ++i)
        assignmentsTest[i] = c.predict(test.getX(i));
    return clusterClassificationAccuracy(
        clusterContingencyMatrix(assignmentsTest, test));
}








double UCB1(double averageValue, int nTries, int totalTries)
    {return averageValue + sqrt(2 * log(totalTries)/nTries);}

template<typename PROBLEM> void TDLearning(PROBLEM& p)
{
    while(p.hasMoreEpisodes())
    {
        double valueCurrent = p.startEpisode();
        while(!p.isInFinalState())
        {
            double valueNext = p.pickNextState();
            p.updateCurrentStateValue(p.learningRate() * (p.reward() +
                p.discountRate() * valueNext - valueCurrent));
            p.goToNextState();
            valueCurrent = valueNext;
        }
        p.updateCurrentStateValue(p.learningRate() *
            (p.reward() - valueCurrent));
    }
}

struct DiscreteValueFunction
{
    Vector<pair<double, int> > values;
    double learningRate(int state){return 1.0/values[state].second;}
    void updateValue(int state, double delta)
    {
        ++values[state].second;
        values[state].first += delta;
    }
    DiscreteValueFunction(int n): values(n, make_pair(0.0, 1)){}
};

struct LinearCombinationValueFunction
{
    Vector<double> weights;
    int n;
    double learningRate(){return 1.0/n;}
    void updateWeights(Vector<double> const& stateFeatures, double delta)
    {//set one of the state features to 1 to have a bias weight
        assert(stateFeatures.getSize() == weights.getSize());
        for(int i = 0; i < weights.getSize(); ++i)
            weights[i] += delta * stateFeatures[i];
        ++n;
    }
    LinearCombinationValueFunction(int theN): weights(theN, 0), n(1) {}
};

struct APriori
{
    LcpTreap<Vector<int>, int> counts;
    int processBasket(Vector<int> const& basket, int round,
        int rPrevMinCount = 0, int r1MinCount = 0)
    {
        int addedCount = 0;
        if(basket.getSize() > round)
        {
            Combinator c(round, basket.getSize());
            do//prepare the current combination of ids, needn't sort if each
            {//basket is already sorted
                Vector<int> key, single;
                for(int i = 0; i < round; ++i) key.append(basket[c.c[i]]);
                quickSort(key.getArray(), key.getSize());
                int* count = counts.find(key);
                if(count) ++*count;//combination is frequent if already
                else if(round == 1)//frequent or round is 1
                {
                    counts.insert(key, 1);
                    ++addedCount;
                }
                else//combination is frequent if the last item and
                {//combination without the last item are both frequent
                    single.append(key.lastItem());
                    if(*counts.find(single) >= r1MinCount)
                    {
                        key.removeLast();
                        if(*counts.find(key) >= rPrevMinCount)
                        {
                            key.append(single[0]);
                            counts.insert(key, 1);
                            ++addedCount;
                        }
                    }
                }
            }while(!c.next());
        }
        return addedCount;
    }
    void noCutProcess(Vector<Vector<int> >const& baskets, int nRounds)
    {
        for(int k = 1; k <= nRounds; ++k)
            for(int i = 0; i < baskets.getSize(); ++i)
                processBasket(baskets[i], k);
    }
};

}//end namespace
#endif

