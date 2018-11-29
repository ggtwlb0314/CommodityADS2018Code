#ifndef STATISTICS_H
#define STATISTICS_H
#include "Random.h"
#include "../Sorting/Sort.h"
#include "../Utils/Bits.h"
#include "../Utils/Vector.h"
#include "../Heaps/Heap.h"
#include "../NumericalMethods/Matrix.h"
#include "../NumericalMethods/NumericalMethods.h"
#include <cstdlib>
namespace igmdk{

template<typename CDF> double invertCDF(CDF const& c, double u,
    double guess = 0, double step0 = 1, double prec = 0.0001)
{
    assert(u > 0 && u < 1);
    auto f = [c, u](double x){return u - c(x);};
    pair<double, double> bracket = findInterval0(f, guess, step0, 100);
    return solveFor0(f, bracket.first, bracket.second, prec).first;
}

double approxErf(double x)
{
    assert(x >= 0);
    double a[6] = {0.0705230784, 0.0422820123, 0.0092705272, 0.0001520143,
        0.0002765672, 0.0000430638}, poly = 1, xPower = x;
    for(int i = 0; i < 6; ++i)
    {
        poly += a[i] * xPower;
        xPower *= x;
    }
    for(int i = 0; i < 4; ++i) poly *= poly;//calculate poly^16
    return 1 - 1/poly;
}
double approxNormalCDF(double x)
{
    double uHalf = approxErf(abs(x)/sqrt(2))/2;
    return 0.5 + (x >= 0 ? uHalf : -uHalf);
}
double approxNormal2SidedConf(double x){return 2 * approxNormalCDF(x) - 1;}

double find2SidedConfZ(double conf)
{
    assert(conf > 0 && conf < 1);
    return invertCDF([](double x){return approxNormalCDF(x);}, 0.5 + conf/2);
}
double find2SidedConfZBonf(int k, double conf = 0.95)
{
    assert(k > 0);
    return find2SidedConfZ(1 - (1 - conf)/k);
}

double findNemenyiSignificantAveRankDiff(int k, int n, bool forControl = false,
    double conf = 0.95)
{//invert rank sum formula
    int nPairs = k * (k + 1)/2;
    double q = sqrt(nPairs/(3.0 * n)),
        z = find2SidedConfZBonf(forControl ? k : nPairs, conf);
    return q * z/n;//for rank average, not sum
}

struct NormalSummary: public ArithmeticType<NormalSummary>
{
    double mean, variance;
    double stddev()const{return sqrt(variance);}
    double error95()const{return 2 * stddev();}
    explicit NormalSummary(double theMean = 0, double theVariance = 0):
        mean(theMean), variance(theVariance){assert(variance >= 0);}
    NormalSummary operator+=(NormalSummary const& b)
    {
        mean += b.mean;
        variance += b.variance;
        return *this;
    }
    NormalSummary operator-=(NormalSummary const& b)
    {
        mean -= b.mean;
        variance += b.variance;
        return *this;
    }
    NormalSummary operator*=(double a)
    {
        mean *= a;
        variance *= a * a;//change f code and description
        return *this;
    }
};

NormalSummary binomialRate(double k, int n)
{
    if(k < 0 || n < 1) return NormalSummary();
    double p = 1.0 * k/n;
    return NormalSummary(p, p * (1 - p)/n);
}
NormalSummary binomialCount(double k, int n)
    {return NormalSummary(k, k < 0 || n < 1 ? 0 : k * (1 - 1.0 * k/n));}

struct IncrementalStatistics
{
    double sum, squaredSum, minimum, maximum;
    long long n;
    IncrementalStatistics(): n(0), sum(0), squaredSum(0),
        minimum(numeric_limits<double>::max()), maximum(-minimum){}
    double getMean()const{return sum/n;}
    double getVariance()const{return n < 2 ? 0 :
        max(0.0, (squaredSum - sum * getMean())/(n - 1.0));}
    double stdev()const{return sqrt(getVariance());}
    void addValue(double x)
    {
        ++n;
        maximum = max(maximum, x);
        minimum = min(minimum, x);
        sum += x;
        squaredSum += x * x;
    }
    NormalSummary getStandardErrorSummary()const
        {return NormalSummary(getMean(), getVariance()/n);}
    double finiteSamplePlusMinusError01(double confidence = 0.95)
    {//empirical Bernstein
        assert(n > 1 && confidence > 0 && confidence < 1);
        double p = 1 - confidence, tempB = log(3/p);
        return (sqrt(2 * (n - 1) * getVariance() * tempB) + 3 * tempB)/n;
    }
    static double bernHelper(double x, double y, double sign)
    {
        return (x + 2 * y + sign * sqrt(x * (x + 4 * y * (1 - y))))/2/
            (1 + x);
    }
    pair<double, double> bernBounds(double confidence = 0.95,
        int nFactor = 1)
    {
        assert(n > 0 && confidence > 0 && confidence < 1);
        double p = 1 - confidence, t = log(2/p)/n, yMin = getMean() - t/3,
            yMax = getMean() + t/3;
        return make_pair(yMin > 1 ? 0 : bernHelper(2 * t, yMin, -1),
            yMax > 1 ? 1 : bernHelper(2 * t, yMax, 1));
    }
};

template<typename FUNCTION> IncrementalStatistics MonteCarloSimulate(
    FUNCTION const& f, long long nSimulations = 1000)
{
    IncrementalStatistics s;
    for(long long i = 0; i < nSimulations; ++i) s.addValue(f());
    return s;
}

pair<double, double> getTConf(IncrementalStatistics const&s, double a = 0.05)
{
    assert(s.n > 1 && a > 0 && a < 1);
    double ste = s.getStandardErrorSummary().stddev() *
        find2SidedConfT(1 - a, s.n - 1);
    return make_pair(s.getMean() - ste, s.getMean() + ste);
}
pair<double, double> getTConf(Vector<double> const& data, double a = 0.05)
{
    IncrementalStatistics s;
    for(int i = 0; i < data.getSize(); ++i) s.addValue(data[i]);
    return getTConf(s, a);
}

//presented in numerical algorithms chapter
double boxVolume(Vector<pair<double, double> > const& box)
{
    double result = 1;
    for(int i = 0; i < box.getSize(); ++i)
        result *= box[i].second - box[i].first;
    return result;
}
struct InsideTrue{
    bool operator()(Vector<double> const& dummy)const{return true;}};
template<typename TEST, typename FUNCTION> pair<double, double>
    MonteCarloIntegrate(Vector<pair<double, double> > const& box, int n,
    TEST const& isInside = TEST(), FUNCTION const& f = FUNCTION())
{
    IncrementalStatistics s;
    for(int i = 0; i < n; ++i)
    {
        Vector<double> point(box.getSize());
        for(int j = 0; j < box.getSize(); ++j)
            point[j] = GlobalRNG().uniform(box[j].first, box[j].second);
        if(isInside(point)) s.addValue(f(point));
    }
    double regionVolume = boxVolume(box) * s.n/n;
    return make_pair(regionVolume * s.getMean(),
        regionVolume * s.getStandardErrorSummary().error95());
}

struct BootstrapResult
{
    double fValue, bias, std, iFa, iF1ma;
    double biasFactor(){return bias/std;}
    pair<double, double> normalInterval(double z = 2)const
        {return make_pair(fValue - z * std, fValue + z * std);}
    pair<double, double> normalBiasAdjustedInterval(double z = 2)const
    {
        pair<double, double> result = normalInterval(z);
        result.first -= bias;
        result.second -= bias;
        return result;
    }
    pair<double, double> pivotalInterval()const
        {return make_pair(2 * fValue - iF1ma, 2 * fValue - iFa);}
};
template<typename BOOTER> BootstrapResult bootstrap(BOOTER& booter,
    int b = 10000, double confidence = 0.95)
{
    assert(b > 2);
    int tailSize = b * (1 - confidence)/2;
    if(tailSize < 1) tailSize = 1;
    if(tailSize > b/2 - 1) tailSize = b/2 - 1;
    //max heap to work with the largest value
    Heap<double, ReverseComparator<double> > left;
    Heap<double> right;//min heap for the smallest value
    double q = booter.eval();
    IncrementalStatistics s;
    for(int i = 0; i < b; ++i)
    {
        booter.boot();//resample
        double value = booter.eval();
        s.addValue(value);
        if(left.getSize() < tailSize)
        {//heaps not full
            left.insert(value);
            right.insert(value);
        }
        else
        {//heaps full - replace if more extreme
            if(value < left.getMin()) left.changeKey(0, value);
            else if(value > right.getMin()) right.changeKey(0, value);
        }
    }
    BootstrapResult r = {q, s.getMean() - q, s.stdev(),
        left.getMin(), right.getMin()};
    return r;
}
template<typename FUNCTION, typename DATA = double> struct BasicBooter
{
    Vector<DATA> const& data;
    Vector<DATA> resample;
    FUNCTION f;
    BasicBooter(Vector<DATA> const& theData, FUNCTION const& theF = FUNCTION())
        :data(theData), f(theF), resample(theData){assert(data.getSize() > 0);}
    void boot()
    {
        for(int i = 0; i < data.getSize(); ++i)
            resample[i] = data[GlobalRNG().mod(data.getSize())];
    }
    double eval()const{return f(resample);}
};

bool confIncludes(pair<double, double> const& interval, double value)
    {return interval.first <= value && value <= interval.second;}

bool isNormal0BestBonf(Vector<NormalSummary> const& data, double aLevel)
{//smallest is best with precision meanPrecision
    int k = data.getSize();
    assert(k > 1);
    double z = find2SidedConfZBonf(k, 1 - aLevel),
        upper = data[0].mean + z * data[0].stddev();
    for(int i = 1; i < k; ++i)
    {
        double lower = data[i].mean - z * data[i].stddev();
        if(lower <= upper) return false;
    }
    return true;
}

template<typename MULTI_FUNCTION> struct OCBA
{
    MULTI_FUNCTION const& f;
    Vector<IncrementalStatistics> data;
    int nDone;
    OCBA(MULTI_FUNCTION const& theF = MULTI_FUNCTION(), int initialSims = 30):
        f(theF), data(theF.getSize())
    {
        int k = f.getSize();
        for(int i = 0; i < k; ++i)
            for(int j = 0; j < initialSims; ++j) data[i].addValue(f(i));
        nDone = k * initialSims;
    }
    pair<Vector<NormalSummary>, int> findBest()
    {
        int k = f.getSize();
        Vector<NormalSummary> s;
        for(int i = 0; i < k; ++i) s.append(data[i].getStandardErrorSummary());
        int bestI = 0, bestRatioI = -1;
        double bestMean = s[0].mean, ratioSum = 0, bestRatio;
        for(int i = 1; i < k; ++i)
            if(s[i].mean < bestMean) bestMean = s[bestI = i].mean;
        swap(s[0], s[bestI]);
        return make_pair(s, bestI);
    }
    void simulateNext()
    {
        pair<Vector<NormalSummary>, int> best = findBest();
        int k = f.getSize(), bestI = best.second, bestRatioI = -1;;
        Vector<NormalSummary> s = best.first;
        //compute the largest OCBA ratio
        double bestMean = s[0].mean, ratioSum = 0, bestRatio;
        for(int i = 1; i < k; ++i)
        {
            double meanDiff = s[i].mean - bestMean, ratio =
                s[i].variance/(meanDiff * meanDiff);
            ratioSum += ratio * ratio/s[i].variance;
            if(bestRatioI == -1 || ratio > bestRatio)
            {
                bestRatio = ratio;
                bestRatioI = i;
            }
        }
        double ratioBest = sqrt(ratioSum * s[0].variance);
        if(ratioBest > bestRatio) bestRatioI = bestI;
        else if(bestRatioI == bestI) bestRatioI = 0;
        //simulate the largest ratio alternative
        data[bestRatioI].addValue(f(bestRatioI));
        ++nDone;
    }
    int simulateTillBest(int simBudget = 100000, double aLevel = 0.05)
    {
        assert(nDone < simBudget);
        int k = f.getSize(), nTests = lgCeiling(simBudget) - lgFloor(nDone);
        while(nDone < simBudget)
        {
            simulateNext();
            if(isPowerOfTwo(nDone) || nDone == simBudget - 1)
            {
                Vector<NormalSummary> s = findBest().first;
                if(isNormal0BestBonf(s, aLevel/nTests)) break;
            }
        }
        return nTests;
    }
};

template<typename FUNCTION> struct SpeedTester
{
    FUNCTION f;
    SpeedTester(FUNCTION const& theFunction = FUNCTION()): f(theFunction){}
    double operator()()const
    {
        int now = clock();
        f();
        return (clock() - now) * 1.0/CLOCKS_PER_SEC;
    }
};

class Sobol
{//SobolPolys do not represent highest and lowest 1s
    enum{B = numeric_limits<double>::digits};
    unsigned long long k;//current simulation number
    Vector<unsigned long long> x, v;//current sample and precomputed data
    double factor;
    int vIndex(int d, int b){return d * B + b;}//to map to array index
public:
    static int maxD(){return 52;}
    Sobol(int d): factor(1.0/twoPower(B)), v(d * B, 0), x(d, 0), k(1)
    {
        assert(d <= maxD());
        unsigned char const SobolPolys[] = {0,1,1,2,1,4,2,4,7,11,13,14,1,13,16,
            19,22,25,1,4,7,8,14,19,21,28,31,32,37,41,42,50,55,56,59,62,14,21,
            22,38,47,49,50,52,56,67,70,84,97,103,115,122},
            SobolDegs[] = {1,2,3,3,4,4,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,7,7,
            7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8};
        for(int i = 0; i < d; ++i)
            for(int j = 0; j < B; ++j)
            {
                unsigned long long value;
                int l = j - SobolDegs[i];
                //j < D case
                if(l < 0) value = (2 * j + 1) * twoPower(B - j - 1);
                else
                {//j >= D case
                    value = v[vIndex(i, l)];
                    value ^= value/twoPower(SobolDegs[i]);
                    for(int k = 1; k < SobolDegs[i]; ++k)
                        if(Bits::get(SobolPolys[i], k - 1))
                            value ^= v[vIndex(i, l + k)];
                }
                v[vIndex(i, j)] = value;
            }
        next();
    }
    void next()
    {
        for(int i = 0, c = rightmost0Count(k++); i < x.getSize(); ++i)
            x[i] ^= v[vIndex(i, c)];
    }
    double getU01Value(int i)const{return x[i] * factor;}
    double getUValue(int i, double a, double b)const
        {return a + (b - a) * getU01Value(i);}
};

class ScrambledSobolHybrid
{
    int D;
    Vector<double> shifts;
    mutable Sobol s;
    Vector<pair<double, double> > box;
public:
    ScrambledSobolHybrid(Vector<pair<double, double> > const& theBox):
        D(theBox.getSize()), shifts(D), s(min(D, Sobol::maxD())), box(theBox)
    {
        for(int i = 0; i < D; ++i)
            shifts[i] = GlobalRNG().uniform(box[i].first, box[i].second);
    }
    Vector<double> operator()()const
    {//first get Sobol variates for the supported dimensions
        Vector<double> next(D);
        for(int i = 0; i < min(D, Sobol::maxD()); ++i)
            next[i] = s.getUValue(i, box[i].first, box[i].second);
        s.next();
        //random for remaining dimensions;
        for(int i = min(D, Sobol::maxD()); i < D; ++i)
            next[i] = GlobalRNG().uniform(box[i].first, box[i].second);
        //scramble
        for(int i = 0; i < D; ++i)
        {
            next[i] += shifts[i] - box[i].first;
            if(next[i] > box[i].second)
                next[i] -= box[i].second - box[i].first;
        }
        return next;
    }
};
template<typename SAMPLER = ScrambledSobolHybrid> class GeometricBoxWrapper
{
    static Vector<pair<double, double> > transformGeometricBox
        (Vector<pair<double, double> > box)
    {
        for(int i = 0; i < box.getSize(); ++i)
        {
            assert(box[i].first > 0 && box[i].first < box[i].second);
            box[i].first = log(box[i].first);
            box[i].second = log(box[i].second);
        }
        return box;
    }
    SAMPLER s;
public:
    GeometricBoxWrapper(Vector<pair<double, double> > const& box):
        s(transformGeometricBox(box)){}
    Vector<double> operator()()const
    {
        Vector<double> result = s();
        for(int i = 0; i < result.getSize(); ++i) result[i] = exp(result[i]);
        return result;
    }
};

template<typename TEST, typename FUNCTION> pair<double, double> SobolIntegrate(
    Vector<pair<double, double> > const& box, int n,
    TEST const& isInside = TEST(), FUNCTION const& f = FUNCTION())
{
    IncrementalStatistics s;
    ScrambledSobolHybrid so(box);
    for(int i = 0; i < n; ++i)
    {
        Vector<double> point = so();
        if(isInside(point)) s.addValue(f(point));
    }
    double regionVolume = boxVolume(box) * s.n/n;
    return make_pair(regionVolume * s.getMean(),
        regionVolume * s.getStandardErrorSummary().error95());
}

template<typename PDF> class GridRWM
{
    PDF f;
    double x, fx, aFrom, aTo;
    int from, to;
    double sampleHelper(double a)
    {
        double xNew = x + GlobalRNG().uniform(-a, a), fxNew = f(xNew);
        if(fx * GlobalRNG().uniform01() <= fxNew)
        {
            x = xNew;
            fx = fxNew;
        }
        return x;
    }
public:
    GridRWM(double x0 = 0, PDF const& theF = PDF(), int from = -10,
        int to = 20): x(x0), f(theF), fx(f(x)), aFrom(pow(2, from)),
        aTo(pow(2, to)) {}
    double sample()
    {
        for(double a = aFrom; a < aTo; a *= 2) sampleHelper(a);
        return x;
    }
};

template<typename PDF> class MultidimGridRWM
{
    PDF f;
    Vector<double> x;
    double fx, aFrom, aTo, factor;
    Vector<double> sampleHelper(double a)
    {
        Vector<double> xNew = x;
        for(int i = 0; i < xNew.getSize(); ++i)
            xNew[i] += GlobalRNG().uniform(-a, a);
        double fxNew = f(xNew);
        if(fx * GlobalRNG().uniform01() <= fxNew)
        {
            x = xNew;
            fx = fxNew;
        }
        return x;
    }
public:
    MultidimGridRWM(Vector<double> const& x0, PDF const& theF = PDF(),
        int from = -10, int to = 20): x(x0), f(theF), fx(f(x)), aFrom(pow(2,
        from)), aTo(pow(2, to)), factor(pow(2, 1.0/x.getSize())) {}
    Vector<double> sample()
    {
        for(double a = aFrom; a < aTo; a *= factor) sampleHelper(a);
        return x;
    }
};

bool normalTestAreEqual(double m1, double v1, double m2, double v2,
    double z = 3)
{
    NormalSummary diff = NormalSummary(m1, v1) - NormalSummary(m2, v2);
    return abs(diff.mean) <= z * diff.stddev();
}

bool signTestAreEqual(double winCount1, double winCount2, double z = 2)
    {return abs(winCount1 - winCount2)/sqrt(winCount1 + winCount2) <= z;}
pair<double, double> countWins(Vector<pair<double, double> > const& data)
{
    int n1 = 0, n2 = 0;
    for(int i = 0; i < data.getSize(); ++i)
    {
        if(data[i].first < data[i].second) ++n1;
        else if(data[i].first > data[i].second) ++n2;
        else{n1 += 0.5; n2 += 0.5;}
    }
    return make_pair(n1, n2);
}
bool signTestPairs(Vector<pair<double, double> > const& data, double z = 2)
{
    pair<double, double> wins = countWins(data);
    return signTestAreEqual(wins.first, wins.second, z);
}

double evaluateChiSquaredCdf(double chi, int n)
{
    assert(chi >= 0 && n > 0);
    double m = 5.0/6 - 1.0/9/n - 7.0/648/n/n + 25.0/2187/n/n/n,
        q2 = 1.0/18/n + 1.0/162/n/n - 37.0/11664/n/n/n, temp = chi/n,
        x = pow(temp, 1.0/6) - pow(temp, 1.0/3)/2 + pow(temp, 1.0/2)/3;
    return approxNormalCDF((x - m)/sqrt(q2));
}
double chiSquaredP(Vector<int> const& counts,
    Vector<double> const& means, int degreesOfFreedomRemoved = 0)
{//
    double chiStat = 0;
    for(int i = 0; i < counts.getSize(); ++i)
    {//enforce 5 in each bin for good approximation
        assert(means[i] >= 5);
        chiStat += (counts[i] - means[i]) * (counts[i] - means[i])/means[i];
    }
    return 1 - evaluateChiSquaredCdf(chiStat,
        counts.getSize() - degreesOfFreedomRemoved);
}

Vector<double> convertToRanks(Vector<double> a)
{//create index array, sort it, and convert indices into ranks
    int n = a.getSize();
    Vector<int> indices(n);
    for(int i = 0; i < n; ++i) indices[i] = i;
    IndexComparator<double> c(a.getArray());
    quickSort(indices.getArray(), 0, n - 1, c);
    for(int i = 0; i < n; ++i)
    {//rank lookahead to scan for ties, then change a entries
        int j = i;
        while(i + 1 < n && c.isEqual(indices[j], indices[i + 1])) ++i;
        double rank = (i + j)/2.0 + 1;
        for(; j <= i; ++j) a[indices[j]] = rank;
    }
    return a;
}

void HolmAdjust(Vector<double>& pValues)
{
    int k = pValues.getSize();
    Vector<int> indices(k);
    for(int i = 0; i < k; ++i) indices[i] = i;
    IndexComparator<double> c(pValues.getArray());
    quickSort(indices.getArray(), 0, k - 1, c);
    for(int i = 0; i < k; ++i) pValues[indices[i]] =  min(1.0, max(i > 0 ?
        pValues[indices[i - 1]] : 0, (k - i) * pValues[indices[i]]));
}

void FDRAdjust(Vector<double>& pValues)
{
    int k = pValues.getSize();
    Vector<int> indices(k);
    for(int i = 0; i < k; ++i) indices[i] = i;
    IndexComparator<double> c(pValues.getArray());
    quickSort(indices.getArray(), 0, k - 1, c);
    for(int i = k - 1; i >= 0; --i) pValues[indices[i]] = min(i < k - 1 ?
        pValues[indices[i + 1]] : 1, pValues[indices[i]] * k/(i + 1));
}

Vector<double> FriedmanRankSums(Vector<Vector<double> > const& a)
{//a[i] is vector of responses on domain i
    assert(a.getSize() > 0 && a[0].getSize() > 1);
    int n = a.getSize(), k = a[0].getSize();
    Vector<double> alternativeRankSums(k);
    for(int i = 0; i < n; ++i)
    {
        assert(a[i].getSize() == k);
        Vector<double> ri = convertToRanks(a[i]);
        for(int j = 0; j < k; ++j) alternativeRankSums[j] += ri[j];
    }
    return alternativeRankSums;
}

double NemenyiAllPairsPValueUnadjusted(double r1, double r2, int n, int k)
    {return 1 - approxNormalCDF(abs(r1 - r2)/sqrt(n * k * (k + 1)/6.0));}
Matrix<double> RankTestAllPairs(Vector<Vector<double> > const& a,
    double NemenyiALevel = 0.05, bool useFDR = false)
{
    Vector<double> rankSums = FriedmanRankSums(a);
    int n = a.getSize(), k = rankSums.getSize();
    Vector<double> temp(k * (k - 1)/2);
    for(int i = 1, index = 0; i < k; ++i) for(int j = 0; j < i; ++j)
        temp[index++] = NemenyiAllPairsPValueUnadjusted(rankSums[i],
            rankSums[j], n, k);
    if(useFDR) FDRAdjust(temp);
    else HolmAdjust(temp);
    Matrix<double> result = Matrix<double>::identity(k);
    for(int i = 1, index = 0; i < k; ++i) for(int j = 0; j < i; ++j)
        result(i, j) = result(j, i) = temp[index++];
    return result;
}

double PearsonCorrelation(Vector<pair<double, double> > const& a)
{
    IncrementalStatistics x, y;
    for(int i = 0; i < a.getSize(); ++i)
    {
        x.addValue(a[i].first);
        y.addValue(a[i].second);
    }
    double covSum = 0;
    for(int i = 0; i < a.getSize(); ++i)
        covSum += (a[i].first - x.getMean()) * (a[i].second - y.getMean());
    return covSum/sqrt(x.getVariance() * y.getVariance());
}

double SpearmanCorrelation(Vector<pair<double, double> > a)
{
    Vector<double> x, y;
    for(int i = 0; i < a.getSize(); ++i)
    {
        x.append(a[i].first);
        y.append(a[i].second);
    }
    x = convertToRanks(x), y = convertToRanks(x);
    for(int i = 0; i < a.getSize(); ++i)
    {
        a[i].first = x[i];
        a[i].second = y[i];
    }
    return PearsonCorrelation(a);
}

double findMaxKSDiff(Vector<double> a, Vector<double> b)
{//helper to calculate max diff
    quickSort(a.getArray(), a.getSize());
    quickSort(b.getArray(), b.getSize());
    double aLevel = 0, bLevel = 0, maxDiff = 0, delA = 1.0/a.getSize(),
        delB = 1.0/b.getSize();
    for(int i = 0, j = 0; i < a.getSize() || j < b.getSize();)
    {
        double x, nextX = numeric_limits<double>::infinity();
        bool useB = i >= a.getSize() || (j < b.getSize() && b[j] < a[i]);
        if(useB)
        {
            x = b[j++];
            bLevel += delB;
        }
        else
        {
            aLevel += delA;
            x = a[i++];
        }//handle equal values--process all before diff update
        if(i < a.getSize() || j < b.getSize())
        {
            useB = i >= a.getSize() || (j < b.getSize() && b[j] < a[i]);
            nextX = useB ? b[j] : a[i];
        }
        if(x != nextX) maxDiff = max(maxDiff, abs(aLevel - bLevel));
    }
    return maxDiff;
}
double KS2SamplePValue(Vector<double> const& a, Vector<double> const& b)
{//calculate the adjustment first, then find p-value of d
    double stddev = sqrt(1.0 * (a.getSize() + b.getSize())/
        (a.getSize() * b.getSize())),
        delta = findMaxKSDiff(a, b)/stddev;
    return 2 * exp(-2 * delta * delta);
}

template<typename CDF> double findMaxKDiff(Vector<double> x, CDF const& cdf)
{//helper to calculate max diff
    quickSort(x.getArray(), x.getSize());
    double level = 0, maxDiff = 0, del = 1.0/x.getSize();
    for(int i = 0; i < x.getSize(); ++i)
    {
        double cdfValue = cdf(x[i]);
        maxDiff = max(maxDiff, abs(cdfValue - level));
        level += del;
        while(i + 1 < x.getSize() && x[i] == x[i + 1])
        {
            level += del;
            ++i;
        }
        maxDiff = max(maxDiff, abs(cdfValue - level));
    }
    return maxDiff;
}
template<typename CDF> double DKWPValue(Vector<double> const& x,
    CDF const& cdf)
{//DKW invalid for p-value < 0.5
    double delta = findMaxKDiff(x, cdf);
    return min(0.5, 2 * exp(-2 * x.getSize() * delta * delta));
}

Vector<double> findSobolIndicesHelper(Vector<double> const& ya,
    Vector<Vector<double> > const& yc)
{//calculate S from YA and YC
    int n = ya.getSize(), D = yc.getSize();
    Vector<double> result(D);
    IncrementalStatistics s;
    for(int i = 0; i < n; ++i) s.addValue(ya[i]);
    double f02 = s.getMean() * s.getMean(), tempa = dotProduct(ya, ya)/n;
    for(int j = 0; j < D; ++j)
        result[j] = max(0.0, (dotProduct(ya, yc[j])/n - f02)/(tempa - f02));
    return result;
}
template<typename FUNCTOR> pair<Vector<double>, Vector<double> >
    findSobolIndicesSaltelli(Vector<pair<Vector<double>, double> > const&
    data, FUNCTOR const& f, int nBoots = 200)
{//calculate ya and yb
    int D = data[0].first.getSize(), n = data.getSize()/2;
    Vector<double> ya(n), yb(n), yaR(n), stdErrs(D);
    for(int i = 0; i < 2 * n; ++i)
        if(i < n) ya[i] = data[i].second;
        else yb[i - n] = data[i].second;
    //calculate yc
    Vector<Vector<double> > yc(D, Vector<double>(n)), ycR = yc;
    for(int j = 0; j < D; ++j)
        for(int i = 0; i < n; ++i)
        {
            Vector<double> x = data[n + i].first;
            x[j] = data[i].first[j];
            yc[j][i] = f(x);
        }
    //bootstrap to find standard deviations
    Vector<IncrementalStatistics> s(D);
    for(int k = 0; k < nBoots; ++k)
    {//resample data rows
        for(int i = 0; i < n; ++i)
        {
            int index = GlobalRNG().mod(n);
            yaR[i] = ya[index];
            for(int j = 0; j < D; ++j) ycR[j][i] = yc[j][index];
        }
        //evaluate
        Vector<double> indicesR = findSobolIndicesHelper(yaR, ycR);
        for(int j = 0; j < D; ++j) s[j].addValue(indicesR[j]);
    }
    for(int j = 0; j < D; ++j) stdErrs[j] = s[j].stdev();
    return make_pair(findSobolIndicesHelper(ya, yc),
        stdErrs * find2SidedConfZBonf(D, 0.95));
}

double trimmedMean(Vector<double> data, double a = 0.2,
    bool isSorted = false)
{
    int n = data.getSize(), trim = a * n;
    assert(n > 0 && a >= 0 && a < 0.5);
    if(!isSorted)
    {
        quickSelect(data.getArray(), n, trim);
        quickSelect(data.getArray(), n, n - trim - 1);
    }
    double sum = 0;
    for(int i = trim; i < n - trim; ++i) sum += data[i];
    return sum/(n - 2 * trim);
}
double trimmedMeanStandardError(Vector<double> data, double a = 0.2,
    bool isSorted = false)
{
    int n = data.getSize(), trim = a * n;
    assert(n > 0 && a >= 0 && a < 0.5);
    if(!isSorted)
    {
        quickSelect(data.getArray(), n, trim);
        quickSelect(data.getArray(), n, n - trim - 1);
    }//Windsorise tails
    for(int i = 0; i < trim; ++i) data[i] = data[trim];
    for(int i = n - trim; i < n; ++i) data[i] = data[n - trim - 1];
    IncrementalStatistics s;//calc regular se of values
    for(int i = 0; i < n; ++i) s.addValue(data[i]);
    return s.getStandardErrorSummary().stddev()/(1 - 2 * a);
}

double quantile(Vector<double> data, double q, bool isSorted = false)
{
    assert(data.getSize() > 0);
    if(q < 0) return -numeric_limits<double>::infinity();
    else if(q > 1) return -numeric_limits<double>::infinity();
    int n = data.getSize(), u = q * n, l = u - 1;
    if(u == n) u = l;//check corner cases
    else if(u == 0 || double(u) != q * n) l = u;//and border values
    if(!isSorted)
    {
        quickSelect(data.getArray(), n, u);
        if(l != u) quickSelect(data.getArray(), u, l);
    }
    return (data[l] + data[u])/2;
}
double median(Vector<double> const& data, bool isSorted = false)
    {return quantile(data, 0.5, isSorted);}
pair<double, double> quantileConf(Vector<double> const& data, double q = 0.5,
    bool isSorted = false, double z = 2)
{
    double d = z * sqrt(q * (1 - q)/data.getSize());
    return make_pair(quantile(data, q - d, isSorted),
        quantile(data, q + d, isSorted));
}

pair<double, double> normal2SampleDiff(double mean1, double ste1,
    double mean2, double ste2, double z)
{//difference of approximately normal-based confidence intervals
    NormalSummary n1(mean1, ste1 * ste1), n2(mean2, ste2 * ste2),
        diff = n1 - n2;
    double ste = diff.stddev() * z;
    return make_pair(diff.mean - ste, diff.mean + ste);
}
pair<double, double> normalConfDiff(double mean1, pair<double, double> const&
    conf1, double mean2, pair<double, double> const& conf2, double z)
{//difference of approximately normal-based confidence intervals
    return normal2SampleDiff(mean1, (conf1.second - conf1.first)/2/z,
        mean2, (conf2.second - conf2.first)/2/z, z);
}
pair<double, double> median2SampleDiffConf(Vector<double> const& samples1,
    Vector<double> const& samples2, double z = 2)
{
    return normalConfDiff(median(samples1), quantileConf(samples1, 0.5,false,
        z), median(samples2), quantileConf(samples2, 0.5, false, z), z);
}

pair<double, double> trimmedMean2SampleDiffConf(Vector<double> const&
    samples1, Vector<double> const& samples2, double z = 2)
{
    return normal2SampleDiff(trimmedMean(samples1),
        trimmedMeanStandardError(samples1), trimmedMean(samples2),
        trimmedMeanStandardError(samples2), z);
}

template<typename PERM_TESTER> double permutationTest(PERM_TESTER& p,
    int b = 10000, int side = 0)//-1 for smaller and 1 for larger
{//"more extreme" means larger for 1-sided
    double f = p();
    IncrementalStatistics left, right;
    for(int i = 0; i < b; ++i)
    {
        p.exchange();
        double fr = p();
        if(side != 1) left.addValue(f <= fr);
        if(side != -1) right.addValue(f >= fr);
    }//90% 2-sided computes 95% 1-sided
    double leftBound = left.bernBounds(0.90).second,
        rightBound = 1 - right.bernBounds(0.90).first;
    return side == 0 ? 2 * min(leftBound, rightBound) :
        side == -1 ? leftBound : rightBound;
}

struct PairedTestMeanPermuter
{
    Vector<pair<double, double> > data;
    double operator()()const
    {
        double sum = 0;
        for(int i = 0; i < data.getSize(); ++i)
            sum += data[i].first - data[i].second;
        return sum;
    }
    void exchange()
    {
        for(int i = 0; i < data.getSize(); ++i) if(GlobalRNG().mod(2))
            swap(data[i].first, data[i].second);
    }
};
double permutationPairedTest(Vector<pair<double, double> > const& data,
    int b = 10000)
{
    PairedTestMeanPermuter p = {data};
    return permutationTest(p, b);
}

double approxTCDF(double t, int v)
{//Hill3 method, always < 10-4?
    assert(v > 0);
    if(v == 1) return 0.5 + atan(t)/PI();//Cauchy
    if(v == 2) return 0.5 + t/2/sqrt(2 + t * t);//also exact
    double a = v - 0.5, b = 48 * a * a, w = sqrt(a * log(1 + t * t/v)), w27[6];
    w27[0] = w * w;
    for(int i = 1; i < 6; ++i) w27[i] = w * w27[i - 1];
    double z = w + (w27[0] + 3) * w/b - (4 * w27[5] + 33 * w27[3] +
        240 * w27[1] + 855 * w)/(10 * b *(b + 0.8 * w27[2] + 100));
    return approxNormalCDF(t > 0 ? z : -z);
}
double approxT2SidedConf(double x, int v){return 2 * approxTCDF(x, v) - 1;}
double find2SidedConfT(double conf, int v)
{
    assert(conf > 0 && conf < 1 && v > 0);
    return invertCDF([v](double x){return approxTCDF(x, v);}, 0.5 + conf/2);
}

}
#endif
