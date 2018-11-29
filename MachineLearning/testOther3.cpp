#include "MachineLearning.h"
#include "MachineLearningOthers.h"
#include "ReadClassificationData.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "../RandomNumberGeneration/Statistics.h"
#include "../ExternalMemoryAlgorithms/File.h"
using namespace igmdk;

void testAPriori()
{
    Vector<Vector<int> > baskets;
    Vector<int> b1, b2, b3, b4;
    b1.append(0);
    b1.append(1);
    b1.append(2);
    b1.append(3);
    baskets.append(b1);
    b2.append(5);
    b2.append(1);
    b2.append(2);
    b2.append(4);
    baskets.append(b2);
    b3.append(7);
    b3.append(1);
    b3.append(2);
    b3.append(6);
    baskets.append(b3);
    b4.append(1);
    b4.append(0);
    b4.append(4);
    b4.append(6);
    baskets.append(b4);
    APriori ap;
    ap.noCutProcess(baskets, 3);
    for(LcpTreap<Vector<int>, int>::Iterator i(ap.counts.begin()); i != ap.counts.end(); ++i)
    {
        for(int j = 0; j < i->key.getSize(); ++j)
        {
            DEBUG(i->key[j]);
        }
        DEBUG(i->value);
    }
}


struct GridWorld
{
    DiscreteValueFunction u;
    int state, nEpisodes, nextState;
    double reward()
    {
        if(state == 3) return 1;
        if(state == 7) return -1;
        return -0.04;
    }
    double discountRate(){return 1;}
    double goToNextState(){state = nextState;}
    double pickNextState()
    {
        int row = state % 4, column = state / 4;
        int rows[4] = {row+1,row,row,row-1};
        int columns[4] = {column,column+1,column-1,column};
        bool set = false;
        for(int i = 0; i < 4; ++i)
        {
            if(rows[i] >= 0 && rows[i] <= 3 && columns[i] >= 0 && columns[i] <= 2 && !(rows[i] == 1 && columns[i] == 1))
            {
                int newState = rows[i] + columns[i] * 4;
                assert(state !=newState);
                if(!set || u.values[newState].first > u.values[nextState].first) {nextState = newState; set = true;}
            }
        }
        assert(set);
        return u.values[nextState].first;
    }
    bool isInFinalState(){return state == 3 || state == 7;}
    double learningRate(){return u.learningRate(state);}
    bool hasMoreEpisodes(){return nEpisodes;}
    double startEpisode()
    {
        do{state = GlobalRNG().mod(12);} while(state == 5);
        --nEpisodes;
        return u.values[state].first;}
    void updateCurrentStateValue(double delta){u.updateValue(state, delta);}
    GridWorld():nEpisodes(100), u(12){}
    void debug()
    {
        for(int i = 0; i < 3; ++i)
        {
            for(int j = 0; j < 4; ++j)
            {
                cout << " " << u.values[j + i * 4].first;
            }
            cout << endl;
        }
    }
};

void testReinforcement()
{
    GridWorld g;
    TDLearning(g);
    g.debug();
}

template<typename DATA> double clusterL2W(DATA const& data,
    Vector<int> const& assignments)
{
    int n = assignments.getSize();
    assert(n > 0);
    int k = valMax(assignments.getArray(), n) + 1;
    Vector<NUMERIC_X> centroids = findCentroids(data, k, assignments);
    Vector<int> sizes(k);
    for(int i = 0; i < n; ++i) ++sizes[assignments[i]];
    double W = 0;
    for(int i = 0; i < n; ++i)
    {
        int c = assignments[i];
        W += sizes[c] * EuclideanDistance<NUMERIC_X>::distanceIncremental(
            data.getX(i), centroids[c]);
    }
    return W;
}
struct ClusterL2WEvaler
{
    template<typename DATA, typename CLUSTERER> double operator()(
        DATA const& data, CLUSTERER const &c, int k)const
        {return log(clusterL2W(data, c(data, k).assignments));}
};
template<typename CLUSTERER, typename EVALER> pair<double, double> clusterL2Gap
    (NUMERIC_X const& minX, NUMERIC_X const& maxX, CLUSTERER const &c, int k,
    int n, EVALER const& e, int B = 20)
{
    IncrementalStatistics s;
    int D = minX.getSize();
    for(int j = 0; j < B; ++j)
    {
        InMemoryData<NUMERIC_X, int> data;
        for(int i = 0; i < n; ++i)
        {
            NUMERIC_X x(D);
            for(int l = 0; l < D; ++l)
                x[l] = GlobalRNG().uniform(minX[l], maxX[l]);
            data.addZ(x, 0);
        }
        s.addValue(e(data, c, k));
    }
    return make_pair(s.getMean(), sqrt(s.getVariance() * (1 + 1.0/B)));
}
template<typename CLUSTERER, typename PARAMS, typename DATA, typename EVALER>
ClusterResult findClustersAndK_GapHelper(CLUSTERER const &c, PARAMS const& p,
    DATA const& data, EVALER const& e, int maxK = -1)
{
    int n = data.getSize();
    if(maxK == -1) maxK = sqrt(n);
    ClusterResult best(Vector<int>(n), -numeric_limits<double>::infinity());
    ScalerMinMax s(data);
    for(int k = 2; k <= maxK; ++k)
    {
        ClusterResult result = c(data, k, p);
        pair<double, double> gap = clusterL2Gap(s.minX, s.maxX, c, k, n, e);
        result.comparableInternalIndex = gap.first - e(data, c, k);
        if(isfinite(result.comparableInternalIndex) &&
           result.comparableInternalIndex - gap.second >
           best.comparableInternalIndex) best = result;
        else break;
    }
    return best;
}
/*copy-paste below member code in k-means to test gap
struct Dummy
{
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
    template<typename DATA> static ClusterResult findClustersAndK_Gap(
        DATA const& data, int maxIterations = 1000, int maxK = -1)
    {
        return findClustersAndK_GapHelper(KMeans(), maxIterations, data,
            ClusterL2WEvaler(), maxK);
    }
    struct GapFunctor
    {
        template<typename DATA> ClusterResult operator()(DATA const& data,
            int k)const
        {
            KMeans km;
            return km(data, k);
        }
        template<typename DATA> ClusterResult operator()(DATA const& data)const
            {return findClustersAndK_Gap(data);}
    };
};*/

template<typename CLUSTERER> int testNumericalClusterer()
{
    DEBUG("Started Reading");
    typedef InMemoryData<NUMERIC_X, int> T;
    Vector<T> dataM(50);//make many enough to avoid ref realloc
    Vector<pair<PermutedData<T>, PermutedData<T> > > data;

    dataM.append(T());
    readIrisData(dataM.lastItem());
    data.append(makeData<T>(dataM));

    /*dataM.append(T());
    readDigitData(dataM.lastItem(), true);
    dataM.append(T());
    readDigitData(dataM.lastItem(), false);
    data.append(makeDataDivided<T>(dataM));

    dataM.append(T());
    readCNEAData(dataM.lastItem());
    data.append(makeData<T>(dataM));
    dataM.append(T());
    readBanknoteData(dataM.lastItem());
    data.append(makeData<T>(dataM));
    dataM.append(T());
    readWDBCData(dataM.lastItem());
    data.append(makeData<T>(dataM));
    dataM.append(T());
    readGlassData(dataM.lastItem());
    data.append(makeData<T>(dataM));
    dataM.append(T());
    readIonosphereData(dataM.lastItem());
    data.append(makeData<T>(dataM));

    dataM.append(T());
    readLetterData(dataM.lastItem());
    data.append(makeData<T>(dataM));

    dataM.append(T());
    readPimaData(dataM.lastItem());
    data.append(makeData<T>(dataM));
    dataM.append(T());
    readSpamData(dataM.lastItem());
    data.append(makeData<T>(dataM));
    dataM.append(T());
    readSpectData(dataM.lastItem(), true);
    dataM.append(T());
    readSpectData(dataM.lastItem(), false);
    data.append(makeDataDivided<T>(dataM));

    dataM.append(T());
    readStatlogData(dataM.lastItem(), true);
    dataM.append(T());
    readStatlogData(dataM.lastItem(), false);
    data.append(makeDataDivided<T>(dataM));

    dataM.append(T());
    readWineData(dataM.lastItem());
    data.append(makeData<T>(dataM));

    dataM.append(T());
    readArceneData(dataM.lastItem(), true);
    dataM.append(T());
    readArceneData(dataM.lastItem(), false);
    data.append(makeDataDivided<T>(dataM));

    dataM.append(T());
    readMadelonData(dataM.lastItem(), true);
    dataM.append(T());
    readMadelonData(dataM.lastItem(), false);
    data.append(makeDataDivided<T>(dataM));*/

    DEBUG("Done Reading");
    int reportNumber = time(0);
    string fPur = "reportPur" + toString(reportNumber) + ".csv";
    string fARand = "reportARand" + toString(reportNumber) + ".csv";
    string fKDiff = "reportKDiff" + toString(reportNumber) + ".csv";
    string fTimer = "reportTimer" + toString(reportNumber) + ".csv";
    string fCAcc = "reportCAcc" + toString(reportNumber) + ".csv";
    string fTAcc = "reportTAcc" + toString(reportNumber) + ".csv";
    string fStab = "reportStab" + toString(reportNumber) + ".csv";
    ++reportNumber;
    for(int i = 0; i < data.getSize(); ++i)
    {
        int start = clock();
        CLUSTERER c;
        //Vector<int> result = c(data[i].first, findNClasses(data[i].first));
        //ScalerMinMax s(data[i].first);
        //ScalerMQ s(data[i].first);
        //ScaledData<PermutedData<T>, ScalerMQ> sd(data[i].first, s);
        ScalerMinMax s(data[i].first);
        ScaledData<PermutedData<T>, ScalerMinMax> sd(data[i].first, s);
        //Vector<int> result = c(sd).assignments;
        Vector<int> result = c(sd, findNClasses(sd)).assignments;

        Matrix<int> counts = clusterContingencyMatrix(result, sd);
        double purity = clusterPurity(counts);
        double aRand = AdjustedRandIndex(counts);
        double relkDiff = (counts.rows - counts.columns)*1.0/counts.columns;
        double cAcc = clusterClassificationAccuracy(counts);

        DEBUG(purity);
        DEBUG(aRand);
        DEBUG(relkDiff);
        DEBUG(cAcc);


        //DEBUG("stability calc - slow!");*/
        /*double stab = findStability(NoParamsClusterer<CLUSTERER>(), EMPTY(), sd, -1);//findNClasses(sd)
        DEBUG(stab);*/
        double timediff = 1.0 * (clock() - start)/CLOCKS_PER_SEC;
        DEBUG(timediff);

        //double tAcc = findTestCAcc<RandomForest>(data[i].first, result, data[i].second);
        //DEBUG(tAcc);

        /*addToCSV(Vector<string>(1, toString(purity)), fPur.c_str());
        addToCSV(Vector<string>(1, toString(aRand)), fARand.c_str());
        addToCSV(Vector<string>(1, toString(relkDiff)), fKDiff.c_str());
        addToCSV(Vector<string>(1, toString(cAcc)), fCAcc.c_str());
        //addToCSV(Vector<string>(1, toString(tAcc)), fTAcc.c_str());
        addToCSV(Vector<string>(1, toString(timediff)), fTimer.c_str());
        //addToCSV(Vector<string>(1, toString(stab)), fStab.c_str());*/
    }
    return 0;
}

void testNumericalClusterers()
{
    /*testNumericalClusterer<SpectralSmart<> >();
    testNumericalClusterer<KMedGeneral<> >();
    testNumericalClusterer<KMeansGeneral>();
    testNumericalClusterer<RKMeansGeneral>();
    testNumericalClusterer<RKMedGeneral<> >();*/
    testNumericalClusterer<EMSmart>();
}

int main(int argc, char *argv[])
{


    /*DDDKMeans();
    testKMeans2();*/
//    DDDHier();
    //testHier();
    /*DDDKMeansProper();
    testKMeansProper2();*/
    //testReinforcement();
    //testAPriori();
    for(int i = 0; i < 1; ++i) testNumericalClusterers();
	return 0;
}


