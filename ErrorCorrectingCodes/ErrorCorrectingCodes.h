#ifndef ERROR_CORRECTING_CODES_H
#define ERROR_CORRECTING_CODES_H
#include "../Utils/Bitset.h"
#include "../HashTable/LinearProbingHashTable.h"
#include "../NumericalMethods/NumericalMethods.h"
#include <cmath>

namespace igmdk{

class CRC32
{
    uint32_t polynomial, constant[256];
public:
    CRC32(uint32_t thePolynomial = 0xFA567D89u):polynomial(thePolynomial)
    {
        for(int i = 0; i < 256; ++i)
        {
            constant[i] = i << 24;//make extended c
            for(int j = 0; j < 8; ++j) constant[i] =
                (constant[i] << 1) ^ (constant[i] >> 31 ? polynomial : 0);
        }
    }
    uint32_t hash(unsigned char* array, int size, uint32_t crc = 0)
    {
        assert(numeric_limits<unsigned char>::digits == 8);
        for(int i = 0; i < size; ++i)
            crc = (crc << 8) ^ constant[(crc >> 24) ^ array[i]];
        return crc;
    }
};

class GF2mArithmetic
{
    int n;
    Vector<int> el2p, p2el;
public:
    int one()const{return 1;}
    int alpha()const{return 2;}
    GF2mArithmetic(int primPoly)
    {//m is the highest set bit of polynomial
        int m = lgFloor(primPoly);
        assert(m <= 16);//avoid using too much memory
        n = twoPower(m);
        el2p = p2el = Vector<int>(n - 1);//0 has no corresponding power
        p2el[0] = 1;//a^0 = 1, a^(n - 1) also 1 so don't store it, and
        //implicitly el2p[1] = 0
        for(int p = 1; p < n - 1; ++p)
        {//calculate a^p from a^(p - 1)
            int e = p2el[p - 1] << 1;//multiply by x
            if(e >= n) e = sub(e, primPoly);//reduce if needed
            el2p[e - 1] = p;
            p2el[p] = e;
        }
    }
    int elementToPower(int x)const
    {
        assert(x > 0 && x < n);
        return el2p[x - 1];
    }
    int powerToElement(int x)const
    {
        assert(x >= 0 && x < n - 1);
        return p2el[x];
    }//both + and - just xor
    int add(int a, int b)const{return a ^ b;}
    int sub(int a, int b)const{return add(a, b);}
    int mult(int a, int b)const
    {//add in power basis and convert back
        return a == 0 || b == 0 ? 0 :
            powerToElement((elementToPower(a) + elementToPower(b)) % (n - 1));
    }
    int div(int a, int b)const
    {//subtract in power basis and convert back
        assert(b != 0);
        return a == 0 ? 0 : powerToElement((elementToPower(a) + (n - 1) -
            elementToPower(b)) % (n - 1));
    }
};

template<typename ITEM, typename ARITHMETIC>
struct Poly: public ArithmeticType<Poly<ITEM, ARITHMETIC> >
{
    Vector<ITEM> storage;
    ARITHMETIC ari;
public:
    int getSize()const{return storage.getSize();}
    int degree()const{return getSize() - 1;}
    Poly(ARITHMETIC const& theAri, Vector<ITEM> const& coefs =//default is 0
         Vector<ITEM>(1, 0)): ari(theAri), storage(coefs)
    {
        assert(getSize() > 0);
        trim();
    }
    static Poly zero(ARITHMETIC const& theAri){return Poly(theAri);}
    ITEM const& operator[](int i)const{return storage[i];}
    void trim()
        {while(getSize() > 1 && storage.lastItem() == 0) storage.removeLast();}
    Poly& operator+=(Poly const& rhs)
    {//add term-by-term, no carry
        while(getSize() < rhs.getSize()) storage.append(0);
        for(int i = 0; i < min(getSize(), rhs.getSize()); ++i)
            storage[i] = ari.add(storage[i], rhs[i]);
        trim();
        return *this;
    }
    Poly& operator-=(Poly const& rhs)
    {//subtract term-by-term, no carry
        while(getSize() < rhs.getSize()) storage.append(0);
        for(int i = 0; i < min(getSize(), rhs.getSize()); ++i)
            storage[i] = ari.sub(storage[i], rhs[i]);
        trim();
        return *this;
    }
    Poly& operator*=(ITEM const& a)
    {
        for(int i = 0; i < getSize(); ++i)
            storage[i] = ari.mult(storage[i], a);
        trim();
        return *this;
    }
    Poly operator*(ITEM const& a)const
    {
        Poly temp(*this);
        temp *= a;
        return temp;
    }
    Poly& operator<<=(int p)
    {
        assert(p >= 0);
        if(p > 0)
        {
            for(int i = 0; i < p; ++i) storage.append(0);
            for(int i = getSize() - 1; i >= p; --i)
            {
                storage[i] = storage[i - p];
                storage[i - p] = 0;
            }
        }
        return *this;
    }
    Poly& operator>>=(int p)
    {
        assert(p >= 0);
        if(p >= getSize()) storage = Vector<ITEM>(1);
        if(p > 0)
        {
            for(int i = 0; i < getSize() - p; ++i) storage[i] = storage[i + p];
            for(int i = 0; i < p; ++i) storage.removeLast();
        }
        return *this;
    }
    Poly& operator*=(Poly const& rhs)
    {//multiply each term of rhs and sum up
        Poly temp(*this);
        *this *= rhs[0];
        for(int i = 1; i < rhs.getSize(); ++i)
        {
            temp <<= 1;
            *this += temp * rhs[i];
        }
        return *this;
    }
    static Poly makeX(ARITHMETIC const& ari)
    {//x = 1 * x + 0 * 1
        Vector<ITEM> coefs(2);
        coefs[1] = ari.one();
        return Poly(ari, coefs);;
    }
    Poly& reduce(Poly const& rhs, Poly& q)
    {//quotient-remainder division, similar to numbers
        assert(rhs.storage.lastItem() != 0 && q == zero(ari));
        Poly one(ari, Vector<ITEM>(1, ari.one()));
        while(getSize() >= rhs.getSize())
        {//field guarantees exact division
            int diff = getSize() - rhs.getSize();
            ITEM temp2 = ari.div(storage.lastItem(), rhs.storage.lastItem());
            assert(storage.lastItem() ==
                ari.mult(temp2, rhs.storage.lastItem()));
            *this -= (rhs << diff) * temp2;
            q += (one << diff) * temp2;
        }
        return *this;
    }
    Poly& operator%=(Poly const& rhs)
    {
        Poly dummyQ(ari);
        return reduce(rhs, dummyQ);
    }
    bool operator==(Poly const& rhs)const
    {
        if(getSize() != rhs.getSize()) return false;
        for(int i = 0; i < getSize(); ++i)if(storage[i] != rhs[i])return false;
        return true;
    }
    ITEM eval(ITEM const& x)const
    {//Horner's algorithm
        ITEM result = storage[0], xpower = x;
        for(int i = 1; i < getSize(); ++i)
        {
            result = ari.add(result, ari.mult(xpower, storage[i]));
            xpower = ari.mult(xpower, x);
        }
        return result;
    }
    Poly formalDeriv()const
    {
        Vector<ITEM> coefs(getSize() - 1);
        for(int i = 0; i < coefs.getSize(); ++i)
            for(int j = 0; j < i + 1; ++j)
                coefs[i] = ari.add(coefs[i], storage[i + 1]);
        return Poly(ari, coefs);
    }

    void debug()
    {
        DEBUG(getSize());
        for(int i = 0; i < getSize(); ++i)
        {
            DEBUG(int(storage[i]));
        }
    }
};

class ReedSolomon
{
    int n, k;
    GF2mArithmetic ari;
    typedef Poly<unsigned char, GF2mArithmetic> P;
    typedef Vector<unsigned char> V;
    P generator;
    pair<P, P> findLocatorAndEvaluator(P const& syndromePoly, int t)const
    {
        P evPrev(ari, V(1, ari.one())), ev = syndromePoly,
            locPrev = P::zero(ari), loc = evPrev;
        evPrev <<= t;
        while(ev.degree() >= t/2)
        {
            P q(ari);
            evPrev.reduce(ev, q);
            swap(ev, evPrev);
            locPrev -= q * loc;
            swap(loc, locPrev);
        }//normalize them
        if(loc != P::zero(ari))
        {
            int normalizer = ari.div(ari.one(), loc[0]);
            loc *= normalizer;
            ev *= normalizer;
        }
        return make_pair(loc, ev);
    }
public:
    ReedSolomon(int theK = 223, int primPoly = 301): ari(primPoly),
        generator(ari, V(1, 1)), k(theK),
        n(twoPower(lgFloor(primPoly)) - 1)
    {
        assert(k < n);
        P x = P::makeX(ari);
        for(int i = 0, aPower = ari.alpha(); i < n - k; ++i)
        {
            generator *= (x - P(ari, V(1, aPower)));
            aPower = ari.mult(aPower, ari.alpha());
        }
        assert(generator.getSize() == n - k + 1);
    }
    V lengthPadBlock(V block)
    {
        assert(block.getSize() < k);
        block.append(block.getSize());
        while(block.getSize() < k) block.append(0);
        return block;
    }
    pair<V, bool> lengthUnpadBlock(V block)
    {
        assert(block.getSize() == k);
        while(block.getSize() >= 0 && block.lastItem() == 0)block.removeLast();
        bool correct = block.getSize() >= 0 &&
            block.lastItem() == block.getSize() - 1;
        assert(correct);
        if(correct) block.removeLast();
        return make_pair(block, correct);
    }
    V encodeBlock(V const& block)const
    {
        assert(block.getSize() == k);
        P c(ari, block);//init c
        c <<= (n - k);//make space for code
        c += c % generator;//add code
        //beware of poly trim if block is 0
        while(c.storage.getSize() < n) c.storage.append(0);
        return c.storage;
    }
    pair<V, bool> decodeBlock(V const& code)const
    {//calculate syndrome polynomial
        assert(code.getSize() == n);
        P c(ari, code);
        int t = n - k, aPower = ari.alpha();
        V syndromes(t);
        for(int i = 0; i < t; ++i)
        {
            syndromes[i] = c.eval(aPower);
            aPower = ari.mult(aPower, ari.alpha());
        }
        P s(ari, syndromes);
        if(s == P::zero(ari))//no error if yes
        {//take out check data and restore trimmed 0's
            c >>= t;
            while(c.storage.getSize() < k) c.storage.append(0);
            return make_pair(c.storage, true);
        }//find locator and evaluator polys
        pair<P, P> locEv = findLocatorAndEvaluator(s, t);
        if(locEv.first == P::zero(ari)) return make_pair(code, false);
        //find locator roots
        V roots;
        for(int i = 1; i < n + 1; ++i)
            if(locEv.first.eval(i) == 0) roots.append(i);
        if(roots.getSize() == 0) return make_pair(code, false);
        //find error values
        P fd = locEv.first.formalDeriv();
        V errors;
        for(int i = 0; i < roots.getSize(); ++i) errors.append(ari.sub(0,
            ari.div(locEv.second.eval(roots[i]), fd.eval(roots[i]))));
        //correct errors
        while(c.storage.getSize() < n) c.storage.append(0);
        for(int i = 0; i < roots.getSize(); ++i)
        {
            int location = ari.elementToPower(ari.div(ari.one(), roots[i]));
            assert(location < c.getSize());
            c.storage[location] = ari.add(c.storage[location], errors[i]);
        }
        if(c % generator != P::zero(ari)) return make_pair(code, false);
        c >>= t;
        return make_pair(c.storage, true);
    }
};

class BooleanMatrix: public ArithmeticType<BooleanMatrix>
{
    int rows, columns;
    int index(int row, int column)const
    {
        assert(row >= 0 && row < rows && column >= 0 && column < columns);
        return row + column * rows;
    }
    Bitset<> items;
public:
    BooleanMatrix(int theRows, int theColumns): rows(theRows),
        columns(theColumns), items(theRows * theColumns)
        {assert(items.getSize() > 0);}
    int getRows()const{return rows;}
    int getColumns()const{return columns;}
    bool operator()(int row, int column)const
        {return items[index(row, column)];}
    void set(int row, int column, bool value = true)
        {items.set(index(row, column), value);}
    BooleanMatrix operator*=(bool scalar)
    {
        if(!scalar) items.setAll(false);
        return *this;
    }
    friend BooleanMatrix operator*(bool scalar, BooleanMatrix const& a)
    {
        BooleanMatrix result(a);
        return result *= scalar;
    }
    friend BooleanMatrix operator*(BooleanMatrix const& a, bool scalar)
        {return scalar * a;}

    BooleanMatrix& operator+=(BooleanMatrix const& rhs)
    {//+ and - are both xor
        assert(rows == rhs.rows && columns == rhs.columns);
        items ^= rhs.items;
        return *this;
    }
    BooleanMatrix& operator-=(BooleanMatrix const& rhs){return *this += rhs;}

    BooleanMatrix& operator*=(BooleanMatrix const& rhs)
    {//the usual row by column
        assert(columns == rhs.rows);
        BooleanMatrix result(rows, rhs.columns);
        for(int i = 0; i < rows; ++i)
            for(int j = 0; j < rhs.columns; ++j)
            {
                bool sum = false;
                for(int k = 0; k < columns; ++k)
                    sum ^= (*this)(i, k) * rhs(k, j);
                result.set(i, j, result(i, j) ^ sum);
            }
        return *this = result;
    }
    Bitset<> operator*(Bitset<> const& v)const
    {//matrix * vector
        assert(columns == v.getSize());
        Bitset<> result(rows);
        for(int i = 0; i < rows; ++i)
            for(int j = 0; j < columns; ++j)
                result.set(i, result[i] ^ ((*this)(i, j) * v[j]));
        return result;
    }//vector * matrix transposed
    friend Bitset<> operator*(Bitset<> const& v, BooleanMatrix const& m)
        {return m.transpose() * v;}

    static BooleanMatrix identity(int n)
    {
        BooleanMatrix result(n, n);
        for(int i = 0; i < n; ++i) result.set(i, i);
        return result;
    }
    BooleanMatrix transpose()const
    {
        BooleanMatrix result(columns, rows);
        for(int i = 0; i < rows; ++i)
            for(int j = 0; j < columns; ++j) result.set(j, i, (*this)(i, j));
        return result;
    }
    bool operator==(BooleanMatrix const& rhs)
    {
        if(rows != rhs.rows || columns != rhs.columns) return false;
        return items == rhs.items;
    }
    void debug()const
    {
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            {
                cout << (*this)(i, j) << " ";
            }
            cout << endl;
        }
    }
};

class LDPC
{
    BooleanMatrix a, g;//sparcity of A not exploited here
    struct H0Functor//for numerical solving for p
    {
        double hValue;
        double H(double p)const{return p > 0 ? p * log2(1/p) : 0;}
        double operator()(double p)const{return H(p) + H(1 - p) - hValue;}
        H0Functor(double theHValue): hValue(theHValue){}
    };
    double pFromCapacity(double capacity)const//solver guaranteed to succeed
        {return solveFor0(H0Functor(1 - capacity), 0, 0.5).first;}
    Bitset<> extractMessage(Bitset<> const &code)const
    {
        int k = getNewK(), n = code.getSize();
        Bitset<> message(k);
        for(int i = 0; i < k; ++i) message.set(i, code[i + n - k]);
        return message;
    }
    unsigned int uIndex(unsigned int r, unsigned int c)const
        {return r * a.getColumns() + c;}
public:
    int getNewK()const{return g.getColumns();}
    LDPC(int n, int k, int wc = 3): a(n - k, n), g(n, n - k)
    {
        int m = n - k, wr = n/(m/wc);
        assert(m % wc == 0 && n % wr == 0 && wc * n == wr * m && m < n);
        //create a
        for(int r = 0; r < m/wc; ++r)//the first section
            for(int c = 0; c < wr; ++c) a.set(r, c + r * wr);
        Vector<int> perm(n);
        for(int c = 0; c < n; ++c) perm[c] = c;
        for(int i = 1; i < wc; ++i)//other sections as permutations of the
        {//first
            GlobalRNG().randomPermutation(perm.getArray(), n);
            for(int r = 0; r < m/wc; ++r)
                for(int c = 0; c < wr; ++c)
                    a.set(r + i * m/wc, perm[c + r * wr]);
        }
        //DEBUG("aOri");
        //a.debug();
        //create H from A
        BooleanMatrix h = a;
        int skip = 0;
        for(int r = 0; r < m; ++r)
        {//find column with 1, if not return
            int cNow = r - skip, c = cNow;
            for(; c < n; ++c) if(h(r, c)) break;
            if(c == n) ++skip;//all-0 row
            else if(c != cNow)//swap columns cNow and c
                for(int rb = 0; rb < m; ++rb)
                {
                    bool temp = h(rb, cNow);
                    h.set(rb, cNow, h(rb, c));
                    h.set(rb, c, temp);
                    //same for a
                    temp = a(rb, cNow);
                    a.set(rb, cNow, a(rb, c));
                    a.set(rb, c, temp);
                }
            for(int rb = 0; rb < m; ++rb)
                if(rb != r && h(rb, cNow))
                    for(c = cNow; c < n; ++c)
                        h.set(rb, c, h(rb, c) ^ h(r, c));
        }
        /*DEBUG("h");
        h.debug();
        DEBUG("a");
        a.debug();*/
        //remove 0 rows from H
        int mProper = m - skip, delta = 0;
        BooleanMatrix hNew(mProper, n);
        for(int r = 0; r < mProper; ++r)
        {//nonzero rows have correct identity part set
            while(!h(r + delta, r) && r < mProper) ++delta;
            for(int c = 0; c < n; ++c) hNew.set(r, c, h(r + delta, c));
        }
        /*DEBUG("h'");
        hNew.debug();*/
        //create g from h
        int kProper = n - mProper;
        g = BooleanMatrix(n, kProper);
        for(int r = 0; r < n; ++r)
            for(int c = 0; c < kProper; ++c)
                if(r < mProper) g.set(r, c, hNew(r, mProper + c));//x part
                else g.set(r, c, r - mProper == c);//identity part
        /*DEBUG("g");
        g.debug();*/
        assert(a * g == BooleanMatrix(m, kProper));
    }
    Bitset<> encode(Bitset<> const& message)const
    {
        assert(message.getSize() == getNewK());
        return g * message;
    }
    pair<Bitset<>, bool> decode(Bitset<> const &code, int maxIter = 1000,
        double p = -1)const
    {
        int n = a.getColumns(), k = getNewK(), m = a.getRows();
        assert(code.getSize() == n && maxIter > 0);
        Bitset<> zero(k), corrected = code;
        if(a * code == zero) return make_pair(extractMessage(code), true);
        if(p == -1) p = pFromCapacity(1.0 * k/n);//find p if not given
        double const llr1 = log((1 - p)/p);//initialize l
        Vector<double> l(n);
        for(int i = 0; i < n; ++i) l[i] = llr1 * (code[i] ? 1 : -1);
        LinearProbingHashTable<unsigned int, double> nu;//initialize nu
        for(int r = 0; r < m; ++r) for(int c = 0; c < n; ++c) if(a(r, c))
            nu.insert(uIndex(r, c), 0);
        while(a * corrected != zero && maxIter-- > 0)//main loop
        {//update nu
            for(int r = 0; r < m; ++r)
            {
                double temp = 1;
                for(int c = 0; c < n; ++c) if(a(r, c))
                    temp *= tanh((*nu.find(uIndex(r, c)) - l[c])/2);
                for(int c = 0; c < n; ++c) if(a(r, c))
                {
                    double *nuv = nu.find(uIndex(r, c)), product = temp/
                        tanh((*nuv - l[c])/2), value = -2 * atanh(product);
                    //set numerical infinities to heuristic 100
                    if(!isfinite(value)) value = 100 * (product > 0 ? -1 : 1);
                    *nuv = value;
                }
            }//update l and the correction
            for(int c = 0; c < n; ++c)
            {
                l[c] = llr1 * (code[c] ? 1 : -1);
                for(int r = 0; r < m; ++r) if(a(r, c))
                    l[c] += *nu.find(uIndex(r, c));
                corrected.set(c, l[c] > 0);
            }
        }
        bool succeeded = maxIter > 0;
        return make_pair(succeeded ? extractMessage(corrected) : code,
            succeeded);
    }
};

}
#endif
