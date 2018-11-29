#include "Misc.h"
#include "MiscAlgsTestAuto.h"
#include <cassert>
#include <iostream>
using namespace std;
using namespace igmdk;

void DDDLRU()
{
    LRUCache<int, int> LRU4_0to9(4);
    for(int i = 0; i < 10; ++i) LRU4_0to9.write(i, i);

    cout << "breakpoint" << endl;
}

int main()
{
    testAllAutoMiscAlgorithms();

    DDDLRU();

    Permutator perm(4);
    for(int j = 0; j < 7; ++j)
    {
        cout << "P" << endl;
        for(int i = 0; i < perm.p.getSize(); ++i) cout << perm.p[i] << " ";
        cout << endl;
        if(perm.next()) break;
    }
    DEBUG("done1");
    perm.advance(0);
    perm.advance(1);
    for(;;)
    {
        cout << "P" << endl;
        for(int i = 0; i < perm.p.getSize(); ++i) cout << perm.p[i] << " ";
        cout << endl;
        if(perm.next()) break;
    }


    DEBUG("done2");

    Combinator comb(4, 6);

    for(int j = 0; j < 7; ++j)
    {
        cout << "C" << endl;
        for(int i = 0; i < comb.c.getSize(); ++i) cout << comb.c[i] << " ";
        cout << endl;
        if(comb.next()) break;
    }
    DEBUG("done3");
    comb.skipAfter(1);
    for(;;)
    {
        cout << "C" << endl;
        for(int i = 0; i < comb.c.getSize(); ++i) cout << comb.c[i] << " ";
        cout << endl;
        if(comb.next()) break;
    }
    DEBUG("done4");
    Partitioner part(4);

    for(;;)
    {
        cout << "Pa" << endl;
        for(int i = 0; i < part.p.getSize(); ++i) cout << part.p[i] << " ";
        cout << endl;
        if(part.next()) break;
    }


	return 0;
}
