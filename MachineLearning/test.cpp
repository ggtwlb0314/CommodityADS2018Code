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

Matrix<double> sampleCostDeterministic(int k)
{
    assert(k > 0);
    int count = 0;
    Matrix<double> result(k, k);
    for(int r = 0; r < k; ++r)
        for(int c = 0; c < k; ++c) if(r != c) result(r, c) = count++ % 2 ? 0.01 : 1;
    scaleCostMatrix(result);
    return result;
}

template<typename LEARNER> int testNumericalClassifier()
{
    DEBUG("Started Reading");
    typedef InMemoryData<NUMERIC_X, int> T;
    Vector<T> dataM(50);//make many enough to avoid ref realloc
    Vector<pair<PermutedData<T>, PermutedData<T> > > data;

    dataM.append(T());
    readIrisData(dataM.lastItem());
    data.append(makeData<T>(dataM));

    dataM.append(T());
    readDigitData(dataM.lastItem(), true);
    dataM.append(T());
    readDigitData(dataM.lastItem(), false);
    data.append(makeDataDivided<T>(dataM));

    /*dataM.append(T());
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
    string fAcc = "reportAcc" + toString(reportNumber) + ".csv";
    string fAveAcc = "reportAveAcc" + toString(reportNumber) + ".csv";
    string fTimer = "reportTimer" + toString(reportNumber) + ".csv";
    string fCount = "reportFeature" + toString(reportNumber) + ".csv";
    string fCost = "reportCost" + toString(reportNumber) + ".csv";
    ++reportNumber;
    for(int i = 0; i < data.getSize(); ++i)
    {
        if(false)//cost
        {
            /*int start = clock();
            int k = findNClasses(data[i].first);
            Matrix<double> c = sampleCostDeterministic(k);
            for(int z = 0; z < k; ++z)
            {
                for(int j = 0; j < k; ++j)
                    cout << c(z, j) << " ";
                cout << endl;
            }
            //LEARNER s(T(data[i].first).data);
            LEARNER s(T(data[i].first).data, c);
            Matrix<int> confusion = evaluateConfusion(evaluateLearner<int>(s, T(data[i].second).data));
            for(int z = 0; z < k; ++z)
            {
                for(int j = 0; j < k; ++j)
                    cout << confusion(z, j) << " ";
                cout << endl;
            }
            ClassifierStats cs(confusion);
            double timediff = 1.0 * (clock() - start)/CLOCKS_PER_SEC;
            double cost = evalConfusionCost(confusion, c);
            DEBUG(cost);
            cs.debug();
            addToCSV(Vector<string>(1, toString(cost)), fCost.c_str());
            addToCSV(Vector<string>(1, toString(timediff)), fTimer.c_str());*/
        }
        else
        {
            int start = clock();
            LEARNER NeuralNetworkIris(data[i].first);
            ClassifierStats cs(evaluateConfusion(evaluateLearner<int>(NeuralNetworkIris, data[i].second)));
            double timediff = 1.0 * (clock() - start)/CLOCKS_PER_SEC;
            cs.debug();

            /*addToCSV(Vector<string>(1, toString(cs.acc.mean)), fAcc.c_str());
            addToCSV(Vector<string>(1, toString(cs.bac.mean)), fAveAcc.c_str());
            addToCSV(Vector<string>(1, toString(timediff)), fTimer.c_str());*/
            //addToCSV(Vector<string>(1, toString(s.model.f.fMap.getSize())), fCount.c_str());
        }
        //system("PAUSE");
    }
    return 0;
}

Matrix<double> getEqualCostMatrix(int nClasses)
{//for testing RMBoost
    Matrix<double> result(nClasses, nClasses);
    for(int i = 0; i < nClasses; ++i)
        for(int j = 0; j < nClasses; ++j) if(i != j) result(i, j) = 1;
    return result;
}
void testNumericalClassifiers()
{
//    testNumericalClassifier<SSVM>();
//    testNumericalClassifier<DecisionTree>();
//
//
//    testNumericalClassifier<SNN>();
    testNumericalClassifier<SLSVM>();
//    testNumericalClassifier<RMBoost<> >();
//    testNumericalClassifier<RandomForest>();
//
//
//    testNumericalClassifier<SImbSVM>();
//
//    testNumericalClassifier<SRaceLSVM>();
//    testNumericalClassifier<SOnlineNN>();
//
//    testNumericalClassifier<ScaledLearner<NoParamsLearner<KNNClassifier<>, int>, int> >();
//
//
//
    //testNumericalClassifier<SimpleBestCombiner>();

    //inactive learners
    //testNumericalClassifier<ScaledLearner<MeanNN<>, int> >();

    //feature selection
    //testNumericalClassifier<SmartFSLearner<> >();

    //cost learning
    //testNumericalClassifier<SBoostedCostSVM>();
    //testNumericalClassifier<SAveCostSVM>();
    //testNumericalClassifier<CostLearner<> >();

}

int main(int argc, char *argv[])
{
    for(int i = 0; i < 1; ++i) testNumericalClassifiers();
	return 0;
}


