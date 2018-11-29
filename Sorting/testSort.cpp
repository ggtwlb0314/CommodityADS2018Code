#include "Sort.h"
#include "SortTestAuto.h"
#include "../Utils/Vector.h"
#include <iostream>
#include <cstdlib>
using namespace igmdk;

template<typename VECTOR, typename COMPARATOR> void multikeyQuicksort(VECTOR*
    vector, int left, int right, COMPARATOR const& c)
{
    if(right - left < 1) return;
    int i, j;
    partition3(vector, left, right, i, j, c);
    ++c.depth;
    multikeyQuicksort(vector, j + 1, i - 1, c);
    --c.depth;
    multikeyQuicksort(vector, left, j, c);
    multikeyQuicksort(vector, i, right, c);
}

void testMultikey()
{//MUST USE POINTER TO ITEM NOT ITEM AND REFLECT THAT IN COMPARATOR!
    string s[7];
    s[0] = "fsdlfjl";
    s[1] = "wejk";
    s[2] = "iosufrwhrew";
    s[3] = "wqjklhdsaiohd";
    s[4] = "wioeurksd";
    s[5] = "";
    s[6] = "w";
    multikeyQuicksortNR(s, 0, 6, VectorComparator<string>());
    for(int i = 0; i < 7 ; ++i)
    {
        cout << s[i] << endl;
    }
    int k = 3;
    multikeyQuickselect(s, 0, 6, k, VectorComparator<string>());
}

int main()
{
    testAllAutoSort();
	testMultikey();
	return 0;
}
