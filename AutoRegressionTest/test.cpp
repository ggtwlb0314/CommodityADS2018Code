#include "../Utils/UtilsTestAuto.h"
#include "../Sorting/SortTestAuto.h"
#include "../RandomTreap/DynamicSortedSequenceTestAuto.h"
#include "../HashTable/HashTableTestAuto.h"
#include "../Heaps/HeapTestAuto.h"
#include "../ExternalMemoryAlgorithms/ExternalMemoryAlgorithmsTestAuto.h"
#include "../StringAlgorithms/StringAlgorithmsTestAuto.h"
#include "../Compression/CompressionTestAuto.h"
#include "../MiscAlgs/MiscAlgsTestAuto.h"
#include "../LargeNumbers/LargeNumberTestAuto.h"
#include "../ComputationalGeometry/ComputationalGeometryTestAuto.h"
#include "../ErrorCorrectingCodes/ErrorCorrectingCodesTestAuto.h"
#include "../Cryptography/CryptographyTestAuto.h"
#include "../NumericalMethods/NumericalMethodsTestAuto.h"

using namespace igmdk;

int main()
{
    DEBUG("All Tests Auto");
    testAllAutoUtils();
    testAllAutoSort();
    testAllAutoDynamicSortedSequence();
    testAllAutoHashTable();
    testAllAutoHeaps();
    testAllAutoMiscAlgorithms();
    testAllAutoExternalMemoryAlgorithms();
    testAllAutoStringAlgorithms();
    testAllAutoCompression();
    testAllAutoComputationalGeometry();
    testAllAutoErrorCorrectingCodes();
    testAllAutoCryptography();
    testAllAutoNumericalMethods();
    DEBUG("All Tests Auto passed");

	return 0;
}
