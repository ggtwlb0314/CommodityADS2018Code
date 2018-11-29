#include "Random.h"
#include "Statistics.h"
#include "../NumericalMethods/NumericalMethods.h"
#include "../NumericalMethods/Matrix.h"
#include "../Utils/Debug.h"
using namespace igmdk;

struct meder
{
    double operator()(Vector<double> observations)const
    {
        return median(observations);
    }
};

struct trimmer
{
    double operator()(Vector<double> observations)const
    {
        return trimmedMean(observations);
    }
};

struct meaner
{
    double operator()(Vector<double> const& observations)const
    {
        double result = observations[0];
        for(int i = 1; i < observations.getSize(); ++i)
        {
            result += observations[i];
        }
        return result / observations.getSize();
    }
};

struct maxer
{
    double operator()(Vector<double> const& observations)const
    {
        double result = observations[0];
        for(int i = 1; i < observations.getSize(); ++i)
        {
            if(observations[i] > result) result = observations[i];
        }
        return result;
    }
};

void testSumHeap()
{
    int N = 1000000;
    SumHeap<double> sh;
    sh.add(0);
	sh.add(1.0/36);
	sh.add(2.0/36);
	sh.add(3.0/36);
	sh.add(4.0/36);
	sh.add(5.0/36);
	sh.add(6.0/36);
	sh.add(5.0/36);
	sh.add(4.0/36);
	sh.add(3.0/36);
	sh.add(2.0/36);
	sh.add(1.0/36);
    int sum = 0;
    for(int i = 0 ; i < N; ++i) sum += sh.next();
    DEBUG(sum*1.0/N);
}

struct XYZ
{
    double operator()()const
    {
        //return GlobalRNG().bernoulli(0.95);
        return GlobalRNG().uniform01();
    }
};

struct BernFunctor
{
    double m, t;
    double operator()(double x)const
    {
        return (m + sqrt(x * (1 - x) * 2 * t) + t/3) - x;
    }
};
pair<double, double> numericalBernBounds(IncrementalStatistics const & si,
    double confidence = 0.95, int nFactor = 1)
{
    int n = si.n;
    assert(n > 0 && confidence > 0 && confidence < 1);
    double p = 1 - confidence, t = log(2/p)/n;
    BernFunctor b = {si.getMean(), t};
    double result = solveFor0(b, 0, 1).first;
    DEBUG(result);
    BernFunctor b2 = {1 - si.getMean(), t};
    double result2 = solveFor0(b2, 0, 1).first;
    DEBUG(result2);
    return make_pair(1 - result2, result);
}
void testIncremental()
{
    XYZ xyz;
    IncrementalStatistics si = MonteCarloSimulate(xyz, 1000);
    NormalSummary s = si.getStandardErrorSummary();
    DEBUG(s.mean);
    DEBUG(s.error95());
    DEBUG(si.finiteSamplePlusMinusError01());
    pair<double, double> bb = si.bernBounds();
    DEBUG(si.getMean() - bb.first);
    DEBUG(bb.second - si.getMean());
    bb = numericalBernBounds(si);
    DEBUG(si.getMean() - bb.first);
    DEBUG(bb.second - si.getMean());

}

struct OCBATest
{
    mutable int nEvals;
    OCBATest(): nEvals(0){}
    int getSize()const{return 6;}
    double operator()(int i)const
    {
        ++nEvals;
        if(i == 0) return GlobalRNG().normal(1, 9);
        if(i == 1) return GlobalRNG().normal(2, 8);
        if(i == 2) return GlobalRNG().normal(3, 7);
        if(i == 3) return GlobalRNG().normal(4, 6);
        if(i == 4) return GlobalRNG().normal(5, 5);
        else return GlobalRNG().normal(6, 4);
    }
};

template<typename MULTI_FUNCTION> int
    simulateSelectBest(MULTI_FUNCTION& f, int n0 = 30, int T = 100000,
    double meanPrecision = 0, double aLevel = 0.05)
{
    int D = f.getSize(), winner = -1;
    assert(D > 1 && n0 > 1 && T > n0 * D);
    Vector<IncrementalStatistics> data(D);
    for(int i = 0; i < D; ++i)
        for(int j = 0; j < n0; ++j) data[i].addValue(f(i));
    int k = n0 * D;
    for(; k < T;)
    {
        Vector<NormalSummary> s;
        for(int i = 0; i < D; ++i) s.append(data[i].getStandardErrorSummary());
        int bestIndex = 0;
        double bestMean = s[0].mean;
        for(int i = 1; i < D; ++i)
            if(s[i].mean < bestMean) bestMean = s[bestIndex = i].mean;
        swap(s[0], s[bestIndex]);
        for(int i = 0; i < D; ++i)
            if(isPowerOfTwo(++k) && isNormal0BestBonf(s, aLevel/lgCeiling(T)))
            {
                return lgCeiling(T);
            }
        for(int i = 0; i < D; ++i) data[i].addValue(f(i));
    }
    return lgCeiling(T);
}
void testOCBA()
{
    IncrementalStatistics sO, sN;
    for(int i = 0; i < 100; ++i)
    {
        OCBATest to;
        OCBA<OCBATest> o(to);
        int nTests = o.simulateTillBest();
        sO.addValue(to.nEvals);
        pair<Vector<NormalSummary>, int> best = o.findBest();
        DEBUG(isNormal0BestBonf(best.first, 0.05/nTests));
        DEBUG(best.second);
        OCBATest t;
        simulateSelectBest(t);
        sN.addValue(t.nEvals);
    }
    DEBUG(sO.getMean());
    DEBUG(sO.getStandardErrorSummary().error95());
    DEBUG(sN.getMean());
    DEBUG(sN.getStandardErrorSummary().error95());
}

struct Normal01SemiPDF
{
    double mean, variance;
    Normal01SemiPDF(double theMean = 100, double theVariance = 10000):
        mean(theMean), variance(theVariance){}
    double operator()(double x)const{x -= mean; return exp(-x * x/2/variance);}
};

struct MultivarNormalSemiPDF
{
    double operator()(Vector<double> x)const
    {
        Matrix<double> b2(3, 3);
        b2(0, 0) = 1;
        b2(0, 1) = 4;
        b2(0, 2) = 5;
        b2(1, 0) = 4;
        b2(1, 1) = 20;
        b2(1, 2) = 32;
        b2(2, 0) = 5;
        b2(2, 1) = 32;
        b2(2, 2) = 64;
        LUP<double> lup(b2);
        Matrix<double> inv = inverse(lup, 3);
        //inv.debug();
        //for(int i = 0; i < x.getSize(); ++i) x[i] -= 100;
        //for(int i = 0; i < x.getSize(); ++i) DEBUG(x[i]);
        //DEBUG(inv * x * x/(-2));
        //DEBUG(exp(inv * x * x/(-2)));
        //system("PAUSE");
        return exp(dotProduct(inv * x, x)/(-2));
    }
};

void testVectorMagicMCMC2()
{
    MultidimGridRWM<MultivarNormalSemiPDF> g(Vector<double>(3, 0.5));
    int n = 10000;
    for(int i = 0; i < n; ++i) g.sample();

    Vector<double> sum(3, 0);
    Matrix<double> outerSum(3, 3);
    for(int i = 0; i < n; ++i)
    {
        Vector<double> x = g.sample();
        sum += x;
        outerSum += outerProduct(x, x);
    }
    Vector<double> mean = sum * (1.0/n);
    for(int i = 0; i < 3; ++i) DEBUG(mean[i]);
    Matrix<double> cov = (outerSum - outerProduct(mean, sum)) * (1.0/(n - 1));
    cov.debug();
}

void testMCMCMagic()
{
    GridRWM<Normal01SemiPDF> s;
    int n = 10;
    for(int i = 0; i < n; ++i) s.sample();
    IncrementalStatistics z;

    for(int i = 0; i < n; ++i) z.addValue(s.sample());
    DEBUG(z.getMean());
    DEBUG(z.getVariance());

}

void testStatTests()
{//should be 0.95 for all
    DEBUG(evaluateChiSquaredCdf(3.84, 1));
    DEBUG(evaluateChiSquaredCdf(11.1, 5));
    DEBUG(evaluateChiSquaredCdf(18.3, 10));
    DEBUG(evaluateChiSquaredCdf(31.4, 20));
    DEBUG(evaluateChiSquaredCdf(124, 100));
}

void testPermPair()
{
    Vector<pair<double, double> > a;
    a.append(pair<double, double>(51.2, 45.8));
    a.append(pair<double, double>(46.5, 41.3));
    a.append(pair<double, double>(24.1, 15.8));
    a.append(pair<double, double>(10.2, 11.1));
    a.append(pair<double, double>(65.3, 58.5));
    a.append(pair<double, double>(92.1, 70.3));
    a.append(pair<double, double>(30.3, 31.6));
    a.append(pair<double, double>(49.2, 35.4));
    DEBUG(permutationPairedTest(a, 10000000));//seems to converge to 0.0236, t-test gives 0.0283
}

struct Uni01CDF
{
    double operator()(double x)const
    {
        if(x <= 0) return 0;
        else if(x >= 1) return 1;
        else return x;
    }
};

void testKS()
{
    Vector<double> a, b;
    /*a.append(-0.15);
    a.append(8.60);
    a.append(5.00);
    a.append(3.71);
    a.append(4.29);
    a.append(7.74);
    a.append(2.48);
    a.append(3.25);
    a.append(-1.15);
    a.append(8.38);

    b.append(2.55);
    b.append(12.07);
    b.append(0.46);
    b.append(0.35);
    b.append(2.69);
    b.append(-0.94);
    b.append(1.73);
    b.append(0.73);
    b.append(-0.35);
    b.append(-0.37);*/

    a.append(-2.18);
    a.append(-1.79);
    a.append(-1.66);
    a.append(-0.65);
    a.append(-0.05);
    a.append(0.54);
    a.append(0.85);
    a.append(1.69);

    b.append(-1.91);
    b.append(-1.22);
    b.append(-0.96);
    b.append(-0.72);
    b.append(0.14);
    b.append(0.82);
    b.append(1.45);
    b.append(1.86);


    DEBUG(KS2SamplePValue(a, b));
}

void DDDAlias()
{
    Vector<double> probabilities;
    for(int i = 0; i < 5; ++i) probabilities.append((i-2)*(i-2)+1);
    normalizeProbs(probabilities);
    AliasMethod alias(probabilities);
    cout << "breakpoint" << endl;
}

void DDDSumHeap()
{
    Vector<double> probabilities;
    for(int i = 0; i < 5; ++i) probabilities.append((i-2)*(i-2)+1);
    normalizeProbs(probabilities);
    SumHeap<double> sumHeap;
    for(int i = 0; i < 5; ++i) sumHeap.add(probabilities[i]);
    cout << "breakpoint" << endl;
}

void testFriedman()
{
    Vector<Vector<double> > responses;
    Vector<double> r1;
    r1.append(14);
    r1.append(23);
    r1.append(26);
    r1.append(30);
    responses.append(r1);
    Vector<double> r2;
    r2.append(19);
    r2.append(25);
    r2.append(25);
    r2.append(33);
    responses.append(r2);
    Vector<double> r3;
    r3.append(17);
    r3.append(22);
    r3.append(29);
    r3.append(28);
    responses.append(r3);
    Vector<double> r4;
    r4.append(17);
    r4.append(21);
    r4.append(28);
    r4.append(27);
    responses.append(r4);
    Vector<double> r5;
    r5.append(16);
    r5.append(24);
    r5.append(28);
    r5.append(32);
    responses.append(r5);
    Vector<double> r6;
    r6.append(15);
    r6.append(23);
    r6.append(27);
    r6.append(36);
    responses.append(r6);
    Vector<double> r7;
    r7.append(18);
    r7.append(26);
    r7.append(27);
    r7.append(26);
    responses.append(r7);
    Vector<double> r8;
    r8.append(16);
    r8.append(22);
    r8.append(30);
    r8.append(32);
    responses.append(r8);
    double k = 4;
    DEBUG("Holm");
    Matrix<double> all = RankTestAllPairs(responses, 0.05, false);
    for(int i = 0; i < k; ++i)
    {
        cout << "-";
        for(int j = 0; j < k; ++j)
        {
            cout << ":" << std::fixed << std::setprecision(4) << all(i, j) << "|";
        }
        cout << endl;
    }
    DEBUG("FDR");
    all = RankTestAllPairs(responses, 0.05, true);
    for(int i = 0; i < k; ++i)
    {
        cout << "-";
        for(int j = 0; j < k; ++j)
        {
            cout << ":" << std::fixed << std::setprecision(4) << all(i, j) << "|";
        }
        cout << endl;
    }
}

struct SobolIndexFunctor
{
    double operator()(Vector<double> const& x)const
    {
        double sum = 0;
        for(int j = 0; j < x.getSize(); ++j) sum += x[j];
        return sum;
    }
};
void testSobolIndex()
{
    int n = 10000, D = 5;
    Vector<pair<Vector<double>, double> > data(n);
    SobolIndexFunctor f;
    for(int i = 0; i < n; ++i)
    {
        Vector<double> x(D);
        double y = 0;
        for(int j = 0; j < D; ++j) x[j] = GlobalRNG().normal01() * j;
        data[i] = make_pair(x, f(x));
    }
    pair<Vector<double>, Vector<double> > result = findSobolIndicesSaltelli(data, f);
    Vector<double> indices = result.first;
    for(int j = 0; j < D; ++j) DEBUG(indices[j]);
    for(int j = 0; j < D; ++j) DEBUG(result.second[j]);
    double estRMSE = 0;
    for(int j = 0; j < D; ++j) estRMSE += result.second[j] * result.second[j];
    DEBUG(sqrt(estRMSE));

    Vector<double> correct(D);
    for(int j = 0; j < D; ++j) correct[j] = 1.0 * j * j;
    normalizeProbs(correct);
    double mse = 0;
    for(int j = 0; j < D; ++j) mse += pow(correct[j] - indices[j], 2);
    DEBUG(sqrt(mse));
}

void testNormalEval()
{
    DEBUG(approxNormal2SidedConf(2));
    DEBUG(find2SidedConfZ(0.95));
    DEBUG(find2SidedConfZBonf(100, 0.95));
}

void testSobol()
{
    DEBUG(Sobol::maxD());
    Sobol so(2);
    //estimate Pi
    double n = 0, nTotal = pow(10, 8);
    for(int i = 0; i < nTotal; ++i)
    {
        double x = so.getU01Value(0), y = so.getU01Value(1);
        if(x * x + y * y <= 1) ++n;
        so.next();
    }
    double pi = 4 * n/nTotal;
    DEBUG(pi);
    double error = 4 * 2 * PI() * log(nTotal) * log(nTotal)/nTotal;
    DEBUG(error);
}

void testTrimmedMean()
{
    Vector<double> data;
    for(int i = 0; i < 10; ++i) data.append(i);
    data[0] = -100;
    data[1] = -200;
    data[8] = 300;
    data[9] = 400;
    double tm = trimmedMean(data);
    DEBUG(tm);
}

void testBootstrap2()
{
    Vector<double> data;
    meaner med;



    IncrementalStatistics s;
    Vector<double> values;
    for(int b = 0; b < 100000; ++b)
    {
        data = Vector<double>();
        for(int i = 0; i < 50; ++i) data.append(GlobalRNG().exponential(1));
        double value = med(data);
        s.addValue(value);
        values.append(value);
    }
    quickSort(values.getArray(), values.getSize());
    DEBUG("exact Monte Carlo");
    DEBUG(s.getMean());
    DEBUG(sqrt(s.getVariance()));
    int b = values.getSize();
    double confidence = 0.95;
    int tailSize = b * (1 - confidence)/2;
    if(tailSize < 1) tailSize = 1;
    if(tailSize > b/2 - 1) tailSize = b/2 - 1;
    DEBUG(values[tailSize]);
    DEBUG(values[b - 1 - tailSize]);
    double bias = s.getMean() - 1;
    DEBUG(bias);

    data = Vector<double>();
    for(int i = 0; i < 50; ++i) data.append(GlobalRNG().exponential(1));
    DEBUG("conf");
    double t = med(data);
    DEBUG(t);
    DEBUG(t + 1 - values[b - 1 - tailSize]);
    DEBUG(t + 1 - values[tailSize]);


    DEBUG("bootstrap");
    BasicBooter<meaner> booter(data);
    DEBUG(booter.eval());
    BootstrapResult r = bootstrap(booter);
    DEBUG(r.fValue);
    DEBUG(r.bias);
    DEBUG(r.biasFactor());
    DEBUG(r.std);
    DEBUG(r.iFa);
    DEBUG(r.iF1ma);
    DEBUG("normal");
    pair<double, double> i = r.normalInterval();
    DEBUG(i.first);
    DEBUG(i.second);
    int left = 0, right = 0;
    for(int j = 0; j < values.getSize(); ++j)
    {
        if(values[j] < t + 1 - i.second) ++left;
        if(values[j] > t + 1 - i.first) ++right;
    }
    DEBUG(left * 1.0/b);
    DEBUG(right * 1.0/b);
    DEBUG((b - left - right) * 1.0/b);

    DEBUG("normalBiasAdjusted");
    i = r.normalBiasAdjustedInterval();
    DEBUG(i.first);
    DEBUG(i.second);

    left = 0, right = 0;
    for(int j = 0; j < values.getSize(); ++j)
    {
        if(values[j] < t + 1 - i.second) ++left;
        if(values[j] > t + 1 - i.first) ++right;
    }
    DEBUG(left * 1.0/b);
    DEBUG(right * 1.0/b);
    DEBUG((b - left - right) * 1.0/b);

    DEBUG("pivotal");
    i = r.pivotalInterval();
    DEBUG(i.first);
    DEBUG(i.second);

    left = 0, right = 0;
    for(int j = 0; j < values.getSize(); ++j)
    {
        if(values[j] < t + 1 - i.second) ++left;
        if(values[j] > t + 1 - i.first) ++right;
    }
    DEBUG(left * 1.0/b);
    DEBUG(right * 1.0/b);
    DEBUG((b - left - right) * 1.0/b);

}

void testBootstrap()
{
    Vector<double> data;
    //data.append(1000);
    //data.append(2000);
    meder med;
    for(int i = 0; i < 1000; ++i) data.append(GlobalRNG().uniform01());
    IncrementalStatistics s;
    for(int i = 0; i < 1000; ++i) s.addValue(data[i]);
    DEBUG(s.getStandardErrorSummary().error95());

    BasicBooter<meder> booter(data);
    BootstrapResult r = bootstrap(booter);
    DEBUG(r.fValue);
    DEBUG(r.bias);
    DEBUG(r.biasFactor());
    DEBUG(r.std);
    DEBUG(r.iFa);
    DEBUG(r.iF1ma);
    DEBUG("normal");
    pair<double, double> i = r.normalInterval();
    DEBUG(i.first);
    DEBUG(i.second);
    DEBUG("normalBiasAdjusted");
    i = r.normalBiasAdjustedInterval();
    DEBUG(i.first);
    DEBUG(i.second);
    DEBUG("pivotal");
    i = r.pivotalInterval();
    DEBUG(i.first);
    DEBUG(i.second);
    DEBUG(r.fValue - i.first);
    DEBUG(i.second - r.fValue);
}

class ImprovedXorshift
{
    uint32_t state1, state2;
    enum{PASSWORD = 19870804};
public:
    ImprovedXorshift(uint32_t seed = time(0) ^ PASSWORD)
    {
        state1 = seed ? seed : PASSWORD;
        state2 = state1;
    }
    uint32_t next()
    {
        state1 ^= state1 << 13;
        state1 ^= state1 >> 17;
        state1 ^= state1 << 5;
        state2 = state2 * 69069U + 1234567U;
        return state1 + state2;
    }//may return 0
    double uniform01(){return 2.32830643653869629E-10 * max(1u, next());}
};

void testGenerators()
{
    //ARC4 x;
    MRG32k3a x;
    x.jumpAhead();
    unsigned long long N = 1 << 3;
    unsigned long long dummy = 0;
    while(N--) dummy += x.next();
    DEBUG(dummy);

    SumHeap<double> st;
    st.add(0.2);
    st.add(0.2);
    st.add(0.2);
    st.add(0.2);
    st.add(0.2);
    DEBUG(st.total());
    DEBUG(st.find(0.1));
    DEBUG(st.find(0.3));
    DEBUG(st.find(0.6));
    DEBUG(st.find(0.9));

    DEBUG(st.find(0));
    DEBUG(st.find(0));
    DEBUG(st.find(0.5));
    DEBUG(st.find(1));

    DEBUG(st.cumulative(0));
    DEBUG(st.cumulative(1));
    DEBUG(st.cumulative(2));
    DEBUG(st.cumulative(3));
    DEBUG(st.cumulative(4));
    DEBUG(st.find(st.cumulative(0)));
    DEBUG(st.find(st.cumulative(1)));
    DEBUG(st.find(st.cumulative(2)));
    DEBUG(st.find(st.cumulative(3)));
    DEBUG(st.find(st.cumulative(4)));
    for(int i = 0; i < 100; ++i)
    {
        DEBUG(st.next());
    }

    clock_t start = clock();
    MersenneTwister random;
    //MersenneTwister64 random;
    //Xorshift random;
    //Xorshift64 random;
    //ImprovedXorshift random;
    //QualityXorshift64 random;
    //ImprovedXorshift64 random;
    unsigned long long sum = 0;
    for(int i = 0; i < 1000000; ++i)
    {
		sum += random.next();
	}
	DEBUG(sum);
	clock_t end = clock();
	int time = (end - start);
    cout << "IX: " << time << endl;

    if(true)
    {
        DEBUG(GlobalRNG().uniform01());
        DEBUG(GlobalRNG().uniform(10, 20));
        DEBUG(GlobalRNG().normal01());
        DEBUG(GlobalRNG().normal(10, 20));
        DEBUG(GlobalRNG().exponential(1));
        DEBUG(GlobalRNG().gamma1(0.5));
        DEBUG(GlobalRNG().gamma1(1.5));
        DEBUG(GlobalRNG().weibull1(20));
        DEBUG(GlobalRNG().erlang(10, 2));
        DEBUG(GlobalRNG().chiSquared(10));
        DEBUG(GlobalRNG().t(10));
        DEBUG(GlobalRNG().logNormal(10, 20));
        DEBUG(GlobalRNG().beta(0.5, 0.5));
        DEBUG(GlobalRNG().F(10 ,20));
        DEBUG(GlobalRNG().cauchy(0, 1));
        DEBUG(GlobalRNG().Levy());
        DEBUG(GlobalRNG().symmetrizedLevy());
        DEBUG(GlobalRNG().binomial(0.7, 20));
        DEBUG(GlobalRNG().geometric(0.7));
        DEBUG(GlobalRNG().poisson(0.7));
		//system("PAUSE");
	}
	int M = 100000;
	double average = 0;
	for(int i = 0; i < M; ++i)
	{
        average += GlobalRNG().beta(0.5, 0.5);
	}
	DEBUG(average/M);

	Vector<NormalSummary> hha;
	NormalSummary n1(10, 100.0/8);
	NormalSummary n2(20, 81.0/8);
	NormalSummary n3(22, 144.0/8);
	hha.append(n1);
	hha.append(n2);
	hha.append(n3);
	DEBUG(isNormal0BestBonf(hha, 0.05));//wont match Chen book - k adjustment

	Vector<NormalSummary> hha2;
	NormalSummary n4(1, 0.1);
	NormalSummary n5(2, 0.2);
	NormalSummary n6(3, 0.3);
	hha2.append(n4);
	hha2.append(n5);
	hha2.append(n6);
	DEBUG(isNormal0BestBonf(hha2, 0.05));
    testVectorMagicMCMC2();
    testMCMCMagic();
}

void testNemenyi()
{
    int k = 10, n = 10;
    DEBUG(findNemenyiSignificantAveRankDiff(1, 1));
    DEBUG(findNemenyiSignificantAveRankDiff(k, n));
}

void testQuantiles()
{
    Vector<double> data;
    for(int i = 0; i < 4; ++i) data.append(i + 1);
    DEBUG(quantile(data, -0.01));
    DEBUG(quantile(data, 0));
    DEBUG(quantile(data, 0.01));
    DEBUG(quantile(data, 0.24));
    DEBUG(quantile(data, 0.25));
    DEBUG(quantile(data, 0.26));
    DEBUG(quantile(data, 0.49));
    DEBUG(quantile(data, 0.5));
    DEBUG(quantile(data, 0.51));
    DEBUG(quantile(data, 0.74));
    DEBUG(quantile(data, 0.75));
    DEBUG(quantile(data, 0.76));
    DEBUG(quantile(data, 0.99));
    DEBUG(quantile(data, 1));
    DEBUG(quantile(data, 1.01));
    for(int i = 4; i < 100; ++i) data.append(i + 1);
    pair<double, double> conf = quantileConf(data, 0.5);
    DEBUG(conf.first);
    DEBUG(median(data));
    DEBUG(conf.second);
    IncrementalStatistics s;
    for(int i = 1; i < 100; ++i) s.addValue(i + 1);
    DEBUG(s.getMean());
    DEBUG(s.getStandardErrorSummary().error95());
    DEBUG(trimmedMean(data));
    DEBUG(2 * trimmedMeanStandardError(data));

    conf = quantileConf(data, 0.01);
    DEBUG(conf.first);
    DEBUG(quantile(data, 0.01));
    DEBUG(conf.second);

    conf = quantileConf(data, 0.99);
    DEBUG(conf.first);
    DEBUG(quantile(data, 0.99));
    DEBUG(conf.second);
}

double approxTCDF2(double t, int v)
{//Gleason method, worst error 2.9e-3 for v = 3 and t = 0.9
    assert(v > 0);
    if(v == 1) return 0.5 + atan(t)/PI();//Cauchy
    if(v == 2) return 0.5 + t/2/sqrt(2 + t * t);//also exact
    double z = sqrt(log(1 + t * t/v)/(v - 1.5 - 0.1/v + 0.5825/v/v)) * (v - 1);
    return approxNormalCDF(t > 0 ? z : -z);
}
void testT()
{//fyi table has limited precision so wont go beyond 5-6 decimals
    DEBUG(approxTCDF(1.96, 3) - 0.975);
    DEBUG(approxTCDF(1.96, 10) - 0.975);
    DEBUG(approxTCDF(1.96, 30) - 0.975);
    DEBUG(approxTCDF(1.96, 100) - 0.975);
    DEBUG(approxTCDF(1.96, 1000) - 0.975);
    DEBUG(approxTCDF(-1.96, 1000) - 0.025);
    DEBUG(approxNormalCDF(-1.96));
    DEBUG(approxNormalCDF(1.96));
    DEBUG(find2SidedConfT(0.95, 3));
    DEBUG(find2SidedConfT(0.95, 10));
    DEBUG(find2SidedConfT(0.95, 30));
    DEBUG(find2SidedConfT(0.95, 100));
    DEBUG(find2SidedConfT(0.95, 1000));

    DEBUG(approxTCDF(12.706, 1) - 0.975);
    DEBUG(approxTCDF2(12.706, 1) - 0.975);
    DEBUG(approxTCDF(4.303, 2) - 0.975);
    DEBUG(approxTCDF2(4.303, 2) - 0.975);
    DEBUG(approxTCDF(0.9, 3) - 0.7828);//worst case for Gleason < 0.0025
    DEBUG(approxTCDF2(0.9, 3) - 0.7828);
    DEBUG(approxTCDF(3.182, 3) - 0.975);
    DEBUG(approxTCDF2(3.182, 3) - 0.975);
    DEBUG(approxTCDF(2.776, 4) - 0.975);
    DEBUG(approxTCDF2(2.776, 4) - 0.975);
    DEBUG(approxTCDF(2.571, 5) - 0.975);
    DEBUG(approxTCDF2(2.571, 5) - 0.975);
    DEBUG(approxTCDF(2.447, 6) - 0.975);
    DEBUG(approxTCDF2(2.447, 6) - 0.975);
    DEBUG(approxTCDF(2.365, 7) - 0.975);
    DEBUG(approxTCDF2(2.365, 7) - 0.975);
    DEBUG(approxTCDF(2.306, 8) - 0.975);
    DEBUG(approxTCDF2(2.306, 8) - 0.975);
    DEBUG(approxTCDF(2.262, 9) - 0.975);
    DEBUG(approxTCDF2(2.262, 9) - 0.975);
    DEBUG(approxTCDF(2.228, 10) - 0.975);
    DEBUG(approxTCDF2(2.228, 10) - 0.975);
}

void testMultivarNormal()
{//code in Matrix.h
    Vector<double> m(2, 0);
    Matrix<double> var = Matrix<double>::identity(2);
    MultivariateNormal mn(m, var);
    Vector<double> sample = mn.next();
    sample.debug();
}

void testChiSquared()
{
    int t = 10000, n = 128;
    assert(isPowerOfTwo(n));
    IncrementalStatistics s;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> means;
        //geometic05
        for(int count = 64; count >= 8; count /= 2)
        {
            means.append(count);
        }
        Vector<int> counts(means.getSize());
        for(int i = 0; i < n; ++i)
        {//power not super good - 0.55 also seems to pass the tests with 128
            //int var = GlobalRNG().geometric(0.55);
            int var = GlobalRNG().geometric05();
            if(var - 1 < counts.getSize()) ++counts[var - 1];//geom starts from 1
        }
        s.addValue(chiSquaredP(counts, means) <= 0.05);
    }
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.05);
}

void testChiSquaredUniform10()
{
    int t = 10000, n = 100;
    IncrementalStatistics s;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> means(10, n/10);
        Vector<int> counts(means.getSize());
        for(int i = 0; i < n; ++i) ++counts[GlobalRNG().mod(counts.getSize())];
        s.addValue(chiSquaredP(counts, means) <= 0.05);
    }
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.05);
}

struct USampler
{
    double operator()()const{return GlobalRNG().uniform01();}
};
template<typename CDF, typename SAMPLER>
void testDKWProportion(int t = 10000, int n = 100)
{
    IncrementalStatistics s;
    CDF c;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples;
        for(int i = 0; i < n; ++i) samples.append(sa());
        s.addValue(DKWPValue(samples, c) <= 0.05);
    }
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.05);
}

struct Norm01CDF
{
    double operator()(double x)const
    {
        return approxNormalCDF(x);
    }
};
struct NSampler
{
    double operator()()const{return GlobalRNG().normal01();}
};
struct T1001CDF
{
    double operator()(double x)const
    {
        return approxTCDF(x, 10);
    }
};
struct T10Sampler
{
    double operator()()const{return GlobalRNG().t(10);}
};
struct Chi10CDF
{
    double operator()(double x)const
    {
        assert(x >= 0);
        return evaluateChiSquaredCdf(x, 10);
    }
};
struct Chi10Sampler
{
    double operator()()const{return GlobalRNG().chiSquared(10);}
};

struct ExpoSampler
{
    double operator()()const{return GlobalRNG().exponential(1);}
};
struct CauchySampler
{
    double operator()()const{return GlobalRNG().cauchy(0, 1);}
};

struct MCMCN01Sampler
{
    struct N01PDF
    {
        double operator()(double x)const{return exp(-x * x/2);}
    };
    mutable GridRWM<N01PDF> s;
    double operator()()const{return s.sample();}
};

template<typename SAMPLER>
void testKS2SampleProportion(int t = 10000, int n = 20, int n2 = 200)
{
    IncrementalStatistics s;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples, samples2;
        for(int i = 0; i < n; ++i) samples.append(sa());
        for(int i = 0; i < n2; ++i) samples2.append(sa());
        s.addValue(KS2SamplePValue(samples, samples2) <= 0.05);
    }
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.05);
}
bool IsDKW2SampleMatch(Vector<double> const& a, Vector<double> const& b,
    double aLevel = 0.05)
{//calculate the adjustment first, then find p-value of d
    double deltaA = sqrt(log(2/aLevel)/(2 * a.getSize())),
        deltaB = sqrt(log(2/aLevel)/(2 * b.getSize()));
    return findMaxKSDiff(a, b) <= deltaA + deltaB;
}
template<typename SAMPLER>
void testDKW2SampleProportion(int t = 10000, int n = 20, int n2 = 200)
{
    IncrementalStatistics s;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples, samples2;
        for(int i = 0; i < n; ++i) samples.append(sa());
        for(int i = 0; i < n2; ++i) samples2.append(sa());
        s.addValue(!IsDKW2SampleMatch(samples, samples2));
    }
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.05);
}

void testDistroMatch()
{//do this for most distros as unit test?
    DEBUG("testChiSquared()");
    testChiSquared();
    DEBUG("testChiSquaredUniform10()");
    testChiSquaredUniform10();
    DEBUG("testDKWProportion<Uni01CDF, USampler>()");
    testDKWProportion<Uni01CDF, USampler>();
    DEBUG("testDKWProportion<Norm01CDF, NSampler>()");
    testDKWProportion<Norm01CDF, NSampler>();
    //ggggggggggggggets under 10% o not perfect sampler
    //DEBUG("testDKWProportion<Norm01CDF, MCMCN01Sampler>()");
    //testDKWProportion<Norm01CDF, MCMCN01Sampler>();
    DEBUG("testDKWProportion<T1001CDF, T10Sampler>()");
    testDKWProportion<T1001CDF, T10Sampler>();
    DEBUG("testDKWProportion<Chi10CDF, Chi10Sampler>()");
    testDKWProportion<Chi10CDF, Chi10Sampler>();
    DEBUG("testKS2SampleProportion<USampler>()");
    testKS2SampleProportion<USampler>();
    //KS always smaller variance than DKW so DKW will lose, particularly for different sizes
    DEBUG("testDKW2SampleProportion<USampler>()");
    testDKW2SampleProportion<USampler>();
}

template<typename SAMPLER>
void testPairedSign(int t = 10000, int n = 100)
{
    IncrementalStatistics s;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<pair<double, double> > samples;
        for(int i = 0; i < n; ++i) samples.append(make_pair(sa(), sa()));
        s.addValue(!signTestPairs(samples));
    }
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.06);//5 fails here
}
//ABOUT T CONF:
//unclear when do use this--for unknown small sample fixed data should use
//trimmed mean most likely or bootstrap BCA but useless estimate anyway
//Belle rule says 12 enough for conf but unclear!?
pair<double, double> tTestPairsP(Vector<pair<double, double> > const& data)
{
    IncrementalStatistics s;
    for(int i = 0; i < data.getSize(); ++i)
    {
        s.addValue(data[i].first - data[i].second);
    }
    return getTConf(s);
}
template<typename SAMPLER>
void testPairedT(int t = 10000, int n = 100)
{
    IncrementalStatistics s;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<pair<double, double> > samples;
        for(int i = 0; i < n; ++i) samples.append(make_pair(sa(), sa()));
        s.addValue(!confIncludes(tTestPairsP(samples), 0));
    }
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.05);//2.58 too small for uniformly most powerful
}
template<typename SAMPLER>
void testPairedPerm(int t = 1000, int n = 100)
{
    IncrementalStatistics s;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<pair<double, double> > samples;
        for(int i = 0; i < n; ++i) samples.append(make_pair(sa(), sa()));
        s.addValue(permutationPairedTest(samples) < 0.05);
    }
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.05);
}
double trimmedTestPairsP(Vector<pair<double, double> > const& data)
{
    Vector<double> diffs;
    for(int i = 0; i < data.getSize(); ++i)
    {
        diffs.append(data[i].first - data[i].second);
    }

    double tm = trimmedMean(diffs), ste = trimmedMeanStandardError(diffs);
    double z = tm/ste;
    return 1 - approxT2SidedConf(z, data.getSize());
    //return 1 - approxTCDF(z, data.getSize());
}
template<typename SAMPLER>
void testPairedTrimmed(int t = 10000, int n = 100)
{
    IncrementalStatistics s;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<pair<double, double> > samples;
        for(int i = 0; i < n; ++i) samples.append(make_pair(sa(), sa()));
        s.addValue(trimmedTestPairsP(samples) < 0.05);
    }
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.05);//2.58 too small for uniformly most powerful
}
void testPairedTests()
{
    testPairedSign<NSampler>();
    testPairedT<NSampler>();
    testPairedTrimmed<NSampler>();
    testPairedPerm<NSampler>();
}

pair<double, double> normal2SampleConf(Vector<double> const& samples1,
    Vector<double> const& samples2, double z = 2)
{
    IncrementalStatistics s1, s2;
    for(int i = 0; i < samples1.getSize(); ++i) s1.addValue(samples1[i]);
    for(int i = 0; i < samples2.getSize(); ++i) s2.addValue(samples2[i]);
    NormalSummary diff = s1.getStandardErrorSummary() -
        s2.getStandardErrorSummary();
    double ste = diff.stddev() * z;
    return make_pair(diff.mean - ste, diff.mean + ste);
}
template<typename SAMPLER>
void test2SampleDiffNormal(double value = 0, int t = 10000, int n = 100)
{
    DEBUG("n");
    IncrementalStatistics s, s2;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples, samples2;
        for(int i = 0; i < n; ++i) samples.append(sa());
        for(int i = 0; i < n; ++i) samples2.append(sa());
        pair<double, double> c = normal2SampleConf(samples, samples2);
        s2.addValue(c.second - c.first);
        s.addValue(!confIncludes(normal2SampleConf(samples, samples2), value));
    }
    DEBUG(s2.getMean());
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.05);
}


template<typename SAMPLER>
void test2SampleDiffTrimmed(double value = 0, double a = 0.05, int t = 10000, int n = 100)
{
    DEBUG("tm");
    IncrementalStatistics s, s2;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples, samples2;
        for(int i = 0; i < n; ++i) samples.append(sa());
        for(int i = 0; i < n; ++i) samples2.append(sa());
        pair<double, double> c = trimmedMean2SampleDiffConf(samples, samples2);
        s2.addValue(c.second - c.first);
        s.addValue(!confIncludes(c, value));
    }
    DEBUG(s2.getMean());
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= a);
}

bool confIntervalIntersect(pair<double, double> const& i1,
    pair<double, double> const& i2)
{//separated if endpoints bounded from each other
    return !(i1.second < i2.first || i2.second < i1.first);
}
pair<double, double> median2SampleConf(Vector<double> const& samples1,
    Vector<double> const& samples2, double z = 2)
{
    pair<double, double> i1 = quantileConf(samples1),
        i2 = quantileConf(samples2);
    return make_pair(i1.first - i2.second, i2.second - i1.first);
}

template<typename SAMPLER>
void test2SampleDiffMedian(double value = 0, int t = 10000, int n = 100)
{
    DEBUG("med");
    IncrementalStatistics s, s2;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples, samples2;
        for(int i = 0; i < n; ++i) samples.append(sa());
        for(int i = 0; i < n; ++i) samples2.append(sa());
        pair<double, double> c = median2SampleConf(samples, samples2);
        s2.addValue(c.second - c.first);
        s.addValue(!confIncludes(median2SampleConf(samples, samples2), value));
    }
    DEBUG(s2.getMean());
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.05);
}

template<typename SAMPLER>
void test2SampleDiffMedianApprox(double value = 0, int t = 10000, int n = 100)
{
    DEBUG("med approx");
    IncrementalStatistics s, s2;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples, samples2;
        for(int i = 0; i < n; ++i) samples.append(sa());
        for(int i = 0; i < n; ++i) samples2.append(sa());
        pair<double, double> c = median2SampleDiffConf(samples, samples2);
        s2.addValue(c.second - c.first);
        s.addValue(!confIncludes(c, value));
    }
    DEBUG(s2.getMean());
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= 0.06);
}

void test2SampleDiff()
{
    DEBUG("uniform");
    test2SampleDiffNormal<USampler>();
    test2SampleDiffMedian<USampler>();
    test2SampleDiffMedianApprox<USampler>();
    test2SampleDiffTrimmed<USampler>();
    DEBUG("normal");
    test2SampleDiffNormal<NSampler>();
    test2SampleDiffMedian<NSampler>();
    test2SampleDiffMedianApprox<NSampler>();
    test2SampleDiffTrimmed<NSampler>();
    DEBUG("expo");
    test2SampleDiffNormal<ExpoSampler>();
    test2SampleDiffMedian<ExpoSampler>();
    test2SampleDiffMedianApprox<ExpoSampler>();
    test2SampleDiffTrimmed<ExpoSampler>();
    DEBUG("Cauchy");
    test2SampleDiffNormal<CauchySampler>();
    test2SampleDiffMedian<CauchySampler>();
    //med approx fails if diff left skew with right skew?
    test2SampleDiffMedianApprox<CauchySampler>();
    //for Caucny trimmed mean diff sensitive to last value? just not as much as mean which is even more sensitive!?
    test2SampleDiffTrimmed<CauchySampler>(0, 0.11); //fails 0.05

    //DO BOOTSRAP ALSO EXTEND IT TO TAKE VECTOR OF SAMPLES?
}

template<typename F>
pair<double, double> bootSampleConf(Vector<double> const& data)
{
    BasicBooter<F> booter(data);
    BootstrapResult r = bootstrap(booter);
    return r.pivotalInterval();
}
template<typename F, typename SAMPLER>
void testSampleBoot(double value = 0, double al=0.05,int t = 10000, int n = 100)
{
    DEBUG("boot");
    IncrementalStatistics s, s2;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples, samples2;
        for(int i = 0; i < n; ++i) samples.append(sa());
        pair<double, double> c = bootSampleConf<F>(samples);
        s2.addValue(c.second - c.first);
        s.addValue(!confIncludes(c, value));
    }
    DEBUG(s2.getMean());
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= al);
}

pair<double, double> normalSampleConf(Vector<double> const& samples1,
    double z = 2)
{
    IncrementalStatistics s1;
    for(int i = 0; i < samples1.getSize(); ++i) s1.addValue(samples1[i]);
    NormalSummary diff = s1.getStandardErrorSummary();
    double ste = diff.stddev() * z;
    return make_pair(diff.mean - ste, diff.mean + ste);
}
template<typename SAMPLER>
void testSampleNormal(double value = 0, double al=0.05,int t = 10000, int n = 100)
{
    DEBUG("n");
    IncrementalStatistics s, s2;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples, samples2;
        for(int i = 0; i < n; ++i) samples.append(sa());
        pair<double, double> c = normalSampleConf(samples);
        s2.addValue(c.second - c.first);
        s.addValue(!confIncludes(c, value));
    }
    DEBUG(s2.getMean());
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= al);
}

pair<double, double> trimmedMeanConf(Vector<double> const& samples1,
    double z = 2)
{
    double tm = trimmedMean(samples1),
        ste = trimmedMeanStandardError(samples1) * z;
    return make_pair(tm - ste, tm + ste);
}
template<typename SAMPLER>
void testSampleTrimmed(double value = 0, double al=0.05,int t = 10000, int n = 100)
{
    DEBUG("tm");
    IncrementalStatistics s, s2;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples, samples2;
        for(int i = 0; i < n; ++i) samples.append(sa());
        pair<double, double> c = trimmedMeanConf(samples);
        s2.addValue(c.second - c.first);
        s.addValue(!confIncludes(c, value));
    }
    DEBUG(s2.getMean());
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= al);
}

template<typename SAMPLER>
void testSampleMedian(double value = 0, double al=0.05,int t = 10000, int n = 100)
{
    DEBUG("med");
    IncrementalStatistics s, s2;
    SAMPLER sa;
    for(int j = 0; j < t; ++j)
    {
        Vector<double> samples, samples2;
        for(int i = 0; i < n; ++i) samples.append(sa());
        pair<double, double> c = quantileConf(samples);
        s2.addValue(c.second - c.first);
        s.addValue(!confIncludes(c, value));
    }
    DEBUG(s2.getMean());
    DEBUG(s.getMean());
    DEBUG(s.bernBounds().first);
    DEBUG(s.bernBounds().second);
    //ok for this to fail?
    //move uncertainly in % estimate to skepticism?
    assert(s.bernBounds().first <= al);
}

void test1SampleConf()
{
    DEBUG("uniform");
    testSampleNormal<USampler>(0.5);
    testSampleBoot<meaner, USampler>(0.5, 0.05, 100);
    testSampleMedian<USampler>(0.5, 0.06);
    testSampleBoot<meder, USampler>(0.5, 0.15, 100);
    testSampleTrimmed<USampler>(0.5);
    testSampleBoot<trimmer, USampler>(0.5, 0.05, 100);
    DEBUG("normal");
    testSampleNormal<NSampler>();
    testSampleBoot<meaner, NSampler>(0, 0.05, 100);
    testSampleMedian<NSampler>(0, 0.06);
    testSampleBoot<meder, NSampler>(0, 0.15, 100);
    testSampleTrimmed<NSampler>();
    testSampleBoot<trimmer, NSampler>(0, 0.05, 100);
    /*DEBUG("expo");
    testSampleNormal<ExpoSampler>();
    testSampleBoot<meaner, ExpoSampler>();
    testSampleMedian<ExpoSampler>();
    testSampleBoot<meder, ExpoSampler>();
    testSampleTrimmed<ExpoSampler>();
    testSampleBoot<trimmer, ExpoSampler>();*/
    DEBUG("Cauchy");
    testSampleNormal<CauchySampler>();
    testSampleBoot<meaner, CauchySampler>(0, 0.05, 100);
    testSampleMedian<CauchySampler>();
    testSampleBoot<meder, CauchySampler>(0, 0.05, 100);
    testSampleTrimmed<CauchySampler>(0, 0.1);
    testSampleBoot<trimmer, CauchySampler>(0, 0.05, 100);

    //DO BOOTSRAP ALSO EXTEND IT TO TAKE VECTOR OF SAMPLES?
}

void testKSEvals()
{
    double corr = exp(-4) - exp(-9);
    DEBUG(corr/exp(-1));
    DEBUG(exp(-16));
    double corr2 = exp(-16) - exp(-25);
}
int main(int argc, char *argv[])
{
    test2SampleDiff();
    return 0;
    test1SampleConf();
    return 0;
    testDistroMatch();
    return 0;
    testPairedTests();
    return 0;
    testKSEvals();
    //return 0;
    testOCBA();
    return 0;
    testMultivarNormal();
    //return 0;
    testGenerators();
    //return 0;
    testT();
    //return 0;
    testQuantiles();
    //return 0;
    testNemenyi();
    //return 0;
    testBootstrap();
    //return 0;
    testBootstrap2();
    //return 0;
    testPermPair();
    //return 0;
    testIncremental();
    //return 0;
    testTrimmedMean();
    //return 0;
    testSobol();
    //return 0;
    testSobolIndex();
    //return 0;
    testNormalEval();
    //return 0;
    testFriedman();
    //return 0;
    DEBUG(approxNormal2SidedConf(2));
    //return 0;
    testKS();
    //return 0;
    DDDAlias();
    DDDSumHeap();

    testStatTests();
    //return 0;
    DEBUG(signTestAreEqual(7, 10, 0.5));
    testSumHeap();
    return 0;
}
