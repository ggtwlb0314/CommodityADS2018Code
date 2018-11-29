#ifndef MISC_H
#define MISC_H
#include "../Utils/Bitset.h"
#include "../Sorting/Sort.h"
#include "../Utils/GCFreelist.h"
#include "../HashTable/LinearProbingHashTable.h"
#include <cmath>

namespace igmdk{

template<typename KEY, typename VALUE,
    typename HASHER = EHash<BUHash> > class LRUCache
{
    typedef KVPair<KEY, VALUE> ITEM;
    typedef SimpleDoublyLinkedList<ITEM> LIST;
    typedef typename LIST::Iterator I;
    LIST l;
    int size, capacity;
    LinearProbingHashTable<KEY, I, HASHER> h;
public:
    LRUCache(int theCapacity): size(0), capacity(theCapacity)
        {assert(capacity > 0);}
    VALUE* read(KEY const& k)
    {
        I* np = h.find(k);
        if(np)
        {//put in front on access
            l.moveBefore(*np, l.begin());
            return &(*np)->value;
        }
        return 0;
    }
    typedef I Iterator;
    Iterator begin(){return l.begin();}
    Iterator end(){return l.end();}
    Iterator evicteeOnWrite(KEY const& k)//none if not full or item in cache
        {return size < capacity || h.find(k) ? end() : l.rBegin();}
    void write(KEY const& k, VALUE const& v)
    {
        VALUE* oldV = read(k);//first check if already inserted
        if(oldV) *oldV = v;//found, update
        else
        {
            Iterator evictee = evicteeOnWrite(k);
            if(evictee != end())
            {
                h.remove(evictee->key);
                l.moveBefore(evictee, l.begin());//recycle evictee
                evictee->key = k;
                evictee->value = v;
            }
            else
            {
                ++size;
                l.prepend(ITEM(k, v));
            }
            h.insert(k, l.begin());
        }
    }
};

template<typename KEY, typename VALUE, typename RESOURCE,
    typename HASHER = EHash<BUHash> > class DelayedCommitLRUCache
{
    RESOURCE& r;
    typedef LRUCache<KEY, pair<VALUE, bool>, HASHER> LRU;
    typedef typename LRU::Iterator I;
    LRU c;
    void commit(I i)
    {
        if(i->value.second) r.write(i->key, i->value.first);
        i->value.second = false;
    }
    void writeHelper(KEY const& k, VALUE const& v, bool fromWrite)
    {//first commit evictee if any
        I i = c.evicteeOnWrite(k);
        if(i != c.end()) commit(i);
        c.write(k, pair<VALUE, bool>(v, fromWrite));
    }
    DelayedCommitLRUCache(DelayedCommitLRUCache const&);//no copying allowed
    DelayedCommitLRUCache& operator=(DelayedCommitLRUCache const&);
public:
    DelayedCommitLRUCache(RESOURCE& theR, int capacity): r(theR), c(capacity)
        {assert(capacity > 0);}
    VALUE const& read(KEY const& k)
    {//first check if in cache
        pair<VALUE, bool>* mv = c.read(k);
        if(!mv)
        {//if not then read from resource and put in cache
            writeHelper(k, r.read(k), false);
            mv = c.read(k);
        }
        return mv->first;
    }
    void write(KEY const& k, VALUE const& v){writeHelper(k, v, true);}
    void flush(){for(I i = c.begin(); i != c.end(); ++i) commit(i);}
    ~DelayedCommitLRUCache(){flush();}
};

class PrimeTable
{
    long long maxN;
    Bitset<> table;//marks odd numbers starting from 3
    long long nToI(long long n)const{return (n - 3)/2;}
public:
    PrimeTable(long long primesUpto): maxN(primesUpto - 1),
        table(nToI(maxN) + 1)
    {
        assert(primesUpto > 1);
        table.setAll(true);
        for(long long i = 3; i <= sqrt(maxN); i += 2)
            if(isPrime(i))//set every odd multiple i <= k <= maxN/i to false
                for(long long k = i; i * k <= maxN; k += 2)
                    table.set(nToI(i * k), false);
    }
    bool isPrime(long long n)const
    {
        assert(n > 0 && n <= maxN);
        return n == 2 || (n > 2 && n % 2 && table[nToI(n)]);
    }
};

struct Permutator
{
    Vector<int> p;
    Permutator(int size){for(int i = 0; i < size; ++i) p.append(i);}
    bool next()
    {//find largest i such that p[i] < p[i + 1]
        int j = p.getSize() - 1, i = j - 1;//start with one-before-last
        while(i >= 0 && p[i] >= p[i + 1]) --i;
        bool backToIdentity = i == -1;
        if(!backToIdentity)
        {//find j such that p[j] is next largest element after p[i]
            while(i < j && p[i] >= p[j]) --j;
            swap(p[i], p[j]);
        }
        p.reverse(i + 1, p.getSize() - 1);
        return backToIdentity;//true if returned to smallest permutation
    }
    bool advance(int i)
    {
        assert(i >= 0 && i < p.getSize());
        quickSort(p.getArray(), i + 1, p.getSize() - 1,
            ReverseComparator<int>());
        return next();
    }
};

struct Combinator
{
    int n;
    Vector<int> c;
    Combinator(int m, int theN): n(theN), c(m, -1)
    {
        assert(m <= n && m > 0);
        skipAfter(0);
    }
    void skipAfter(int i)
    {//increment c[i], and reset all c[j] for j > i
        assert(i >= 0 && i < c.getSize());
        ++c[i];
        for(int j = i + 1; j < c.getSize(); ++j) c[j] = c[j - 1] + 1;
    }
    bool next()
    {//find rightmost c[i] which can be increased
        int i = c.getSize() - 1;
        while(i >= 0 && c[i] == n - c.getSize() + i) --i;
        bool finished = i == -1;
        if(!finished) skipAfter(i);
        return finished;
    }
};

struct Partitioner
{
    Vector<int> p;
    Partitioner(int n): p(n, 0) {assert(n > 0);}
    bool skipAfter(int k)
    {//set trailing elements to maximum values and call next
        assert(k >= 0 && k < p.getSize());
        for(int i = k; i < p.getSize(); ++i) p[i] = i;
        return next();
    }
    bool next()
    {//find rightmost p[j] which can be increased
        int m = 0, j = -1;
        for(int i = 0; i < p.getSize(); ++i)
        {
            if(p[i] < m) j = i;
            m = max(m, p[i] + 1);
        }
        bool finished = j == -1;
        if(!finished)
        {//increase it and reset the tail
            ++p[j];
            for(int i = j + 1; i < p.getSize(); ++i) p[i] = 0;
        }
        return finished;
    }
};

}
#endif
