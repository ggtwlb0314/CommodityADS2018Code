#ifndef DYNAMIC_SORTED_SEQUENCE_TEST_AUTO_H
#define DYNAMIC_SORTED_SEQUENCE_TEST_AUTO_H

#include "../HashTable/MapTestAutoHelper.h"
#include "SkipList.h"
#include "Treap.h"
#include "LcpTreap.h"
#include "Trie.h"

namespace igmdk{

template<typename KEY> struct DefaultRank
{
    unsigned char* array;
    int size;
	DefaultRank(KEY const& key)
	{//works with pod types only
		array = (unsigned char*)&key;
		size = sizeof(key);
	}
};

template<typename KEY, typename ITEM, typename RANK = DefaultRank<KEY> >
struct TrieWrapper
{
	TernaryTreapTrie<ITEM> trie;
	ITEM* find(KEY const& key)
	{
		RANK rank(key);
		return trie.find(rank.array, rank.size);
	}
	void insert(KEY const& key, ITEM const& item)
	{
		RANK rank(key);
		trie.insert(rank.array, rank.size, item);
	}
	void remove(KEY const& key)
	{
		RANK rank(key);
		trie.remove(rank.array, rank.size);
	}
};

template<typename DSS_II> void testNthElementHelper(int n = 100000)
{
    DSS_II m;
	for(int i = 0; i < n; ++i) m.insert(i, -i);

	typename DSS_II::Iterator iter = m.begin();
	for(int i = 0; i < n; ++i)
    {
        assert(iter != m.end());
        assert(iter->key == m.nthElement(i)->key);
        ++iter;
    }
}

void testSkipListAuto()
{
    DEBUG("testSkipListAuto");
    testMapAutoHelper<SkipList<int, int> >();
    DEBUG("testSkipListAuto passed");
}

void testTreapAuto()
{
    DEBUG("testTreapAuto");
    testMapAutoHelper<Treap<int, int> >();
    testNthElementHelper<Treap<int, int> >();
    DEBUG("testTreapAuto passed");
}

template<typename ITEM> struct AsVectorSize1Comparator
{
    bool operator()(ITEM const& lhs, ITEM const& rhs, int i)const
    {
        return lhs < rhs;
    }
    bool isEqual(ITEM const& lhs, ITEM const& rhs, int i)const
    {
        return lhs == rhs;
    }
    bool isEqual(ITEM const& lhs, ITEM const& rhs)const
    {
        return lhs == rhs;
    }
    bool operator()(ITEM const& lhs, ITEM const& rhs)const
    {
        lhs < rhs;
    }
    int getSize(ITEM const& value)const{return 1;}
};
void testLcpTreapAuto()
{
    DEBUG("testLcpTreapAuto");
    testMapAutoHelper<LcpTreap<int, int, AsVectorSize1Comparator<int> > >();
    testNthElementHelper<LcpTreap<Fat2, int> >();
    DEBUG("testLcpTreapAuto passed");
}

void testTTTAuto()
{
    DEBUG("testTTTAuto");
    TrieWrapper<int, int> m;
    int n = 100000;
	for(int i = 0; i < n; ++i) m.insert(i, -i);
    for(int i = 0; i < n; ++i)
    {
        assert(m.find(i));
        assert(*m.find(i) == -i);
        m.remove(i);
        assert(!m.find(i));
    }
    DEBUG("testTTTAuto passed");
}

void testAllAutoDynamicSortedSequence()
{
    DEBUG("testAllAutoDynamicSortedSequence");
    testSkipListAuto();
    testTreapAuto();
    testLcpTreapAuto();
    testTTTAuto();
}

}//end namespace
#endif
