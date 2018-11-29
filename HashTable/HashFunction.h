#ifndef HASH_FUNCTION_H
#define HASH_FUNCTION_H
#include <string>
#include "../RandomNumberGeneration/Random.h"
#include "../Utils/Bits.h"
namespace igmdk{

class PrimeHash
{
    static uint32_t const PRIME = (1ull << 32) - 5;
    uint32_t seed;
public:
    PrimeHash(): seed(GlobalRNG().next() % PRIME) {}
    typedef uint32_t WORD_TYPE;
    unsigned long long max()const{return PRIME - 1;}
    unsigned long long operator()(uint32_t x)const
        {return (unsigned long long)seed * x % PRIME;}
    class Builder
    {
        unsigned long long sum;
        uint32_t a;
        friend PrimeHash;
        Builder(uint32_t theSeed): sum(0), a(theSeed) {}
    public:
        void add(uint32_t xi)
        {//unlikely possible overflow from adding but that's ok
            sum += (unsigned long long)a * xi;
            a = Xorshift::transform(a);
        }
    };
    Builder makeBuilder()const{return Builder(seed);}
    unsigned long long operator()(Builder b)const{return b.sum % PRIME;}
};

class PrimeHash2
{
    static uint32_t const PRIME = (1ull << 32) - 5;
    uint32_t seed;
public:
    PrimeHash2(): seed(GlobalRNG().next() % PRIME) {}
    typedef uint32_t WORD_TYPE;
    unsigned long long max()const{return PRIME - 1;}
    unsigned long long operator()(uint32_t x)const
        {return (unsigned long long)seed * x % PRIME;}
    class Builder
    {
        unsigned long long sum;
        uint32_t seed;
        friend PrimeHash2;
        Builder(uint32_t theSeed): sum(0), seed(theSeed) {}
    public://unlikely possible overflow from add & mult but that's ok
        void add(uint32_t xi){sum = seed * (sum + xi) % PRIME;}
    };
    Builder makeBuilder()const{return Builder(seed);}
    unsigned long long operator()(Builder b)const{return b.sum;}
};

template<typename HASHER> class MHash
{
    unsigned long long m;
    HASHER h;
public:
    MHash(unsigned long long theM): m(theM)
        {assert(theM > 0 && theM <= h.max());}
    typedef typename HASHER::WORD_TYPE WORD_TYPE;
    unsigned long long max()const{return m - 1;}
    template<typename POD> unsigned long long operator()(POD const& x)const
        {return h(x) % m;}
    typedef typename HASHER::Builder Builder;
    Builder makeBuilder()const{return h.makeBuilder();}
    unsigned long long operator()(Builder b)const{return h(b) % m;}
};
template<typename HASHER> class BHash
{
    unsigned long long mask;
    HASHER h;
public:
    BHash(unsigned long long m): mask(m - 1){assert(m > 0 && isPowerOfTwo(m));}
    typedef typename HASHER::WORD_TYPE WORD_TYPE;
    unsigned long long max()const{return mask;}
    template<typename POD> unsigned long long operator()(POD const& x)const
        {return h(x) & mask;}
    typedef typename HASHER::Builder Builder;
    Builder makeBuilder()const{return h.makeBuilder();}
    unsigned long long operator()(Builder b)const{return h(b) & mask;}
};

class BUHash
{
    uint32_t a, wLB;
    BHash<PrimeHash> h;
public:
    BUHash(unsigned long long m): a(GlobalRNG().next() | 1),//ensure non-0
        wLB(32 - lgCeiling(m)), h(m) {assert(m > 0 && isPowerOfTwo(m));}
    typedef uint32_t WORD_TYPE;
    unsigned long long max()const{return h.max();}
    uint32_t operator()(uint32_t const& x)const
        {return (a * x) >> wLB;}
    typedef BHash<PrimeHash>::Builder Builder;
    Builder makeBuilder()const{return h.makeBuilder();}
    unsigned long long operator()(Builder b)const{return h(b);}
};

template<typename HASHER> class EHash
{//takes special case to avoid template substitution compile errors
    HASHER h;
    template<typename WORD> unsigned long long hash(WORD x, true_type)const
    {//integral type - hash as word if possible
        if(sizeof(WORD) <= sizeof(WORD_TYPE)) return h(x);
        return hash(x, false_type());//or as POD
    }//hash non-integral as array of size 1
    template<typename POD> unsigned long long hash(POD x, false_type)const
        {return operator()(&x, 1);}
public:
    EHash(){}//for h that use no m
    EHash(unsigned long long m): h(m) {}
    typedef typename HASHER::WORD_TYPE WORD_TYPE;
    unsigned long long max()const{return h.max();}
    template<typename POD> unsigned long long operator()(POD x)const
        {return hash(x, is_integral<POD>());}//select integral vs not hash
    class Builder
    {
        enum{K = sizeof(WORD_TYPE)};
        union
        {
            WORD_TYPE xi;
            unsigned char bytes[K];
        };
        int byteIndex;
        typename HASHER::Builder b;
        friend EHash;
        Builder(EHash const& eh): xi(0), byteIndex(0), b(eh.h.makeBuilder()) {}
        template<typename WORD> void add(WORD const& xi, true_type)
        {
            if(sizeof(WORD) % sizeof(WORD_TYPE) == 0)
                //if exact multiple, add as word_type
                for(int i = 0; i < sizeof(WORD)/sizeof(WORD_TYPE); ++i)
                    b.add(((WORD_TYPE*)&xi)[i]);
            else add(xi, false_type());//else as POD
        }
        template<typename POD> void add(POD const& xi, false_type)
        {//is_trivially_copyable is optionally supported - uncomment when is
            //assert(is_trivially_copyable<POD>::value);//cast must work
            for(int i = 0; i < sizeof(xi); ++i) add(((unsigned char*)&xi)[i]);
        }
    public:
        void add(unsigned char bi)
        {
            bytes[byteIndex++] = bi;
            if(byteIndex >= K)
            {
                byteIndex = 0;
                b.add(xi);
                xi = 0;
            }
        }
        template<typename POD> void add(POD const& xi)
            {add(xi, is_integral<POD>());}
        typename HASHER::Builder operator()()
        {//finalize remaining xi if any
            if(byteIndex > 0) b.add(xi);
            return b;
        }
    };
    Builder makeBuilder()const{return Builder(*this);}
    unsigned long long operator()(Builder b)const{return h(b());}
    template<typename POD>
    unsigned long long operator()(POD* array, int size)const
    {
        Builder b(makeBuilder());
        for(int i = 0; i < size; ++i) b.add(array[i]);
        return operator()(b);
    }
};

class TableHash
{//works with M and B, no need for E
    enum{N = 1 << numeric_limits<unsigned char>::digits};
    unsigned int table[N];
public:
    TableHash(){for(int i = 0; i < N; ++i) table[i] = GlobalRNG().next();}
    typedef unsigned char WORD_TYPE;
    unsigned long long max()const{return numeric_limits<unsigned int>::max();}
    template<typename POD> unsigned int operator()(POD const& x)const
    {
        Builder b(makeBuilder());
        for(int i = 0; i < sizeof(x); ++i) b.add(((unsigned char*)&x)[i]);
        return b.sum;
    }
    unsigned int update(unsigned int currentHash, unsigned char byte)
        const{return currentHash ^ table[byte];}//for both add and remove
    class Builder
    {
        unsigned long long sum;
        TableHash const& h;
        friend TableHash;
        Builder(TableHash const& theH): sum(0), h(theH) {}
    public://unlikely possible overflow from add & mult but that's ok
        void add(unsigned char xi){sum ^= h.table[xi];}
    };
    Builder makeBuilder()const{return Builder(*this);}
    unsigned long long operator()(Builder b)const{return b.sum;}
};

struct FNVHash
{//works with M and B, no need for E
    typedef unsigned char WORD_TYPE;
    unsigned long long max()const{return numeric_limits<uint32_t>::max();}
    template<typename POD> uint32_t operator()(POD const& x)const
    {
        Builder b(makeBuilder());
        for(int i = 0; i < sizeof(x); ++i) b.add(((unsigned char*)&x)[i]);
        return b.sum;
    }
    class Builder
    {
        uint32_t sum;
        friend FNVHash;
        Builder(): sum(2166136261u) {}
    public://unlikely possible overflow from add & mult but that's ok
        void add(unsigned char xi){sum = (sum * 16777619) ^ xi;}
    };
    Builder makeBuilder()const{return Builder();}
    uint32_t operator()(Builder b)const{return b.sum;}
};

struct FNVHash64
{//works with M and B, no need for E
    typedef unsigned char WORD_TYPE;
    unsigned long long max()const{return numeric_limits<uint64_t>::max();}
    template<typename POD> uint64_t operator()(POD const& x)const
    {
        Builder b(makeBuilder());
        for(int i = 0; i < sizeof(x); ++i) b.add(((unsigned char*)&x)[i]);
        return b.sum;
    }
    class Builder
    {
        uint64_t sum;
        friend FNVHash64;
        Builder(): sum(14695981039346656037ull) {}
    public:
        void add(unsigned char xi){sum = (sum * 1099511628211ull) ^ xi;}
    };
    Builder makeBuilder()const{return Builder();}
    uint64_t operator()(Builder b)const{return b.sum;}
};

struct XorshiftHash
{
    typedef uint32_t WORD_TYPE;
    unsigned long long max()const{return numeric_limits<uint32_t>::max();}
    uint32_t operator()(uint32_t x)const{return Xorshift::transform(x);}
    class Builder
    {
        uint32_t sum;
        friend XorshiftHash;
        Builder(): sum(0) {}
    public:
        void add(uint32_t xi){sum = Xorshift::transform(sum + xi);}
    };
    Builder makeBuilder()const{return Builder();}
    uint32_t operator()(Builder b)const{return b.sum;}
};
struct Xorshift64Hash
{
    typedef uint64_t WORD_TYPE;
    unsigned long long max()const{return numeric_limits<uint64_t>::max();}
    uint64_t operator()(uint64_t x)const
        {return QualityXorshift64::transform(x);}
    class Builder
    {
        uint64_t sum;
        friend Xorshift64Hash;
        Builder(): sum(0) {}
    public:
        void add(uint64_t xi){sum = QualityXorshift64::transform(sum + xi);}
    };
    Builder makeBuilder()const{return Builder();}
    uint64_t operator()(Builder b)const{return b.sum;}
};

class FairHash
{//works with M and B, not E
    unsigned int operator()(unsigned char* array, int size)const
    {
        unsigned int result = 0;
        for(int i = 0; i < min<int>(size, sizeof(unsigned int)); ++i)
            if(array[i]) result = (result << 8) | array[i];
        return result;
    }
    unsigned int operator()(unsigned int x, true_type)const{return x;}
    template<typename POD> unsigned int operator()(POD const& x, false_type)
        const{return operator()((unsigned char*)&x, sizeof(x));}
public:
    typedef unsigned int WORD_TYPE;//use word type here
    unsigned long long max()const{return numeric_limits<unsigned int>::max();}
    template<typename POD> unsigned int operator()(POD x)const
        {return operator()(x, is_integral<POD>());}
    typedef EMPTY Builder;
};

template<typename HASHER = EHash<BUHash> > class DataHash
{
    HASHER h;
public:
    DataHash(unsigned long long m): h(m){}
    typedef typename HASHER::WORD_TYPE WORD_TYPE;
    unsigned long long max()const{return h.max();}
    unsigned long long operator()(string const& item)const
        {return h(item.c_str(), item.size());}
    template<typename VECTOR> unsigned long long operator()(VECTOR const& item)
        const{return h(item.getArray(), item.getSize());}
    typedef EMPTY Builder;
};

}
#endif
