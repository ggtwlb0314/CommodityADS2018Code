#include "ChainingHashTable.h"
#include "LinearProbingHashTable.h"
#include "BloomFilter.h"
#include "HashTableTestAuto.h"
#include "../RandomNumberGeneration/Statistics.h"
#include "../ExternalMemoryAlgorithms/CSV.h"
#include <iostream>
#include <cmath>
#include <functional>
using namespace igmdk;

template<typename T> void timeRT(int N)
{
    T t;
	for(int i = 0; i < N; ++i)
	{
		t.insert(i,i);
	}
	for(int j = 0; j < 5; ++j)
	{
		for(int i = 0; i < N; ++i)
		{
			assert(t.find(i));
			assert(*t.find(i) == i);
			//t.remove(i);
		}
	}
}

template<typename H> struct FunctionTesterCI
{
    void operator()()const
    {
        timeRT<ChainingHashTable<int, int, H> >(1000000);
    }
};
template<typename H> struct FunctionTesterLI
{
    void operator()()const
    {
        timeRT<LinearProbingHashTable<int, int, H> >(1000000);
    }
};
template<typename H> struct FunctionTesterCF
{
    void operator()()const
    {
        timeRT<ChainingHashTable<Fat2, int, H> >(100000);
    }
};
template<typename H> struct FunctionTesterLF
{
    void operator()()const
    {
        timeRT<LinearProbingHashTable<Fat2, int, H> >(100000);
    }
};

template<typename H> double testInt(H const& h)
{
    int now = clock();
    unsigned int sum = 0;
    for(int i = 0; i < 1000000000; ++i)//
	{
		sum += h(i);
	}
	DEBUG(sum);
	return (clock() - now) * 1.0/CLOCKS_PER_SEC;
}

template<typename H> double testFat10(H const& h)
{
    int now = clock();
    unsigned int sum = 0;
    for(int i = 0; i < 100000000; ++i)//
	{
		sum += h(Fat2(i));
	}
	DEBUG(sum);
	return (clock() - now) * 1.0/CLOCKS_PER_SEC;
}

template<typename H> void testSpeedHHelper(string const& name, H const& h,
    Vector<Vector<string> >& matrix)
{
    Vector<string> titles, row;
    DEBUG(name);
    titles.append("Hasher");
    row.append(name);
    double intSpeed = testInt(h);
    DEBUG(intSpeed);
    titles.append("Int");
    row.append(toString(intSpeed));
    Vector<std::function<void(void)> > functors;
    Vector<string> names;
    functors.append(FunctionTesterCI<H>());
    names.append("Ch");
    functors.append(FunctionTesterLI<H>());
    names.append("LP");
    double fat10Speed = testFat10(h);
    DEBUG(fat10Speed);
    titles.append("Fat10");
    row.append(toString(fat10Speed));
    functors.append(FunctionTesterCF<H>());
    names.append("Ch");
    functors.append(FunctionTesterLF<H>());
    names.append("LP");
    for(int i = 0; i < functors.getSize(); ++i)
    {
        DEBUG(names[i]);
        IncrementalStatistics si = MonteCarloSimulate(
            SpeedTester<std::function<void(void)> >(functors[i]), 100);
        DEBUG(si.getMean());
        titles.append(names[i]);
        row.append(toString(si.getMean()));
        DEBUG(si.error95());
        DEBUG(si.minimum);
        DEBUG(si.maximum);
        titles.append("+-");
        titles.append("Min");
        titles.append("Max");
        row.append(toString(si.error95()));
        row.append(toString(si.minimum));
        row.append(toString(si.maximum));
    }
    if(matrix.getSize() == 0) matrix.append(titles);
    matrix.append(row);
}

void sillyTest()
{
    EHash<BHash<PrimeHash> > h(64);
    //EHash<BHash<FairHash> > h(64);
    for(int i = 0; i < 100; ++i)
    {
        DEBUG(h(i));
        DEBUG(i & 63);
    }
    int m = twoPower(20);
    DEBUG(testFat10(EHash<BUHash>(m)));
    DEBUG(testInt(EHash<BHash<PrimeHash> >(m)));
    DEBUG(testFat10(EHash<BHash<PrimeHash> >(m)));
    DEBUG(testFat10(EHash<BHash<PrimeHash2> >(m)));

}

void testSpeedH()
{
    Vector<Vector<string> > matrix;
    int m = twoPower(20);
    testSpeedHHelper("E-BU", EHash<BUHash>(m), matrix);
    testSpeedHHelper("E-B-Prime", EHash<BHash<PrimeHash> >(m), matrix);
    testSpeedHHelper("E-B-Prime2", EHash<BHash<PrimeHash2> >(m), matrix);
    testSpeedHHelper("B-FNV", BHash<FNVHash>(m), matrix);
    testSpeedHHelper("M-FNV", MHash<FNVHash>(m), matrix);
    testSpeedHHelper("B-FNV64>", BHash<FNVHash>(m), matrix);
    testSpeedHHelper("E-B-X", EHash<BHash<XorshiftHash> >(m), matrix);
    testSpeedHHelper("E-B-X64", EHash<BHash<Xorshift64Hash> >(m), matrix);
    testSpeedHHelper("BHash<FairHash>", BHash<FairHash>(m), matrix);
    testSpeedHHelper("E-B-Table", EHash<BHash<TableHash> >(m), matrix);
    createCSV(matrix, "TestResults.csv");
}

void DDDChaining()
{
    ChainingHashTable<int, int> chainingH0to9;
    for(int i = 0; i < 10; ++i)
	{
		chainingH0to9.insert(i, i);
	}
    cout << "breakpoint" << endl;
}

void DDDLinearProbing()
{
    LinearProbingHashTable<int, int> linearProbingH0to9;
    for(int i = 0; i < 10; ++i)
	{
		linearProbingH0to9.insert(i, i);
	}
    cout << "breakpoint" << endl;
}

void DDDBloomFilter()
{
    BloomFilter<int> bF16_3_0to9(16, 3);
    for(int i = 0; i < 10; ++i)
	{
		bF16_3_0to9.insert(i);
	}
}

int main()
{
    testAllAutoHashTable();
    return 0;
    sillyTest();
    return 0;
    testSpeedH();
    return 0;
    DDDChaining();
    DDDLinearProbing();
    DDDBloomFilter();
	return 0;
}
