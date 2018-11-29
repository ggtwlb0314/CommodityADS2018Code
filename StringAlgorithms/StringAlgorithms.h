#ifndef STRING_ALGORITHMS_H
#define STRING_ALGORITHMS_H
#include "../Utils/Utils.h"
#include "../Utils/Vector.h"
#include "../Utils/GCFreelist.h"
#include "../Utils/Stack.h"
#include "../Utils/Bits.h"
#include "../Graphs/Graph.h"
namespace igmdk{

template<typename VECTOR, typename VECTOR2> bool matchesAt(int position,
    VECTOR2 text, VECTOR pattern, int patternSize)
{//allows different text and pattern types
    int i = 0;
    while(i < patternSize && pattern[i] == text[i + position]) ++i;
    return i == patternSize;
}

struct HorspoolHash
{//ignore size parameter - the matcher will have enough table size
    HorspoolHash(int dummy){}
    struct Builder
    {
        unsigned char c;
        void add(unsigned char theC){c = theC;}
    };
    Builder makeBuilder(){return Builder();}
    unsigned char operator()(Builder b){return b.c;}
};
template<typename VECTOR, typename HASHER = HorspoolHash>
class HashQ
{
    enum{CHAR_ALPHABET_SIZE = 1 << numeric_limits<unsigned char>::digits};
    int patternSize, q;
    Vector<int> shifts;//size is a power of 2 for fast hashing
    VECTOR const &pattern;
    HASHER h;
    typedef typename HASHER::Builder B;
public:
    HashQ(VECTOR const& thePattern, int thePatternSize, int theQ = 1): q(theQ),
        pattern(thePattern), patternSize(thePatternSize), shifts(max<int>(
        CHAR_ALPHABET_SIZE, nextPowerOfTwo(ceiling(patternSize, q)))),
        h(shifts.getSize())
    {//precompute shifts
        assert(patternSize >= q);
        int temp = patternSize - q;
        for(int i = 0; i < shifts.getSize(); ++i) shifts[i] = temp + 1;
        for(int i = 0; i < temp; ++i)
        {
            B b(h.makeBuilder());
            for(int j = 0; j < q; ++j) b.add(pattern[i + j]);
            shifts[h(b)] = temp - i;
        }
    }//return match position and next start position of (-1, -1)
    template<typename VECTOR2> pair<int, int> findNext(VECTOR2 const& text,
        int textSize, int start = 0)//allow different text and pattern types
    {
        while(start + patternSize <= textSize)
        {
            int result = start, hStart = start + patternSize - q;
            B b(h.makeBuilder());
            for(int j = 0; j < q; ++j) b.add(text[hStart + j]);
            start += shifts[h(b)];
            if(matchesAt(result, text, pattern, patternSize))
                return make_pair(result, start);
        }
        return make_pair(-1, -1);
    }
};

template<typename VECTOR, typename HASHER = HorspoolHash> class WuManber
{
    enum{CHAR_ALPHABET_SIZE = 1 << numeric_limits<unsigned char>::digits};
    Vector<pair<VECTOR, int> > const& patterns;
    Vector<int> shifts;//size is a power of 2 for fast hashing
    Vector<Vector<int> > candidates;
    int q, minPatternSize;
    HASHER h;
    typedef typename HASHER::Builder B;
public:
    WuManber(Vector<pair<VECTOR, int> > const& thePatterns, int theQ = 1,
        double avePatternSize = 1): q(theQ), patterns(thePatterns), shifts(
        max<int>(CHAR_ALPHABET_SIZE, nextPowerOfTwo(avePatternSize *
        patterns.getSize()/q))), candidates(shifts.getSize()),
        h(shifts.getSize()), minPatternSize(numeric_limits<int>::max())
    {//precompute shifts
        for(int i = 0; i < patterns.getSize(); ++i)
            minPatternSize = min(patterns[i].second, minPatternSize);
        assert(patterns.getSize() > 0 && minPatternSize >= q);
        int temp = minPatternSize - q;
        for(int i = 0; i < shifts.getSize(); ++i) shifts[i] = temp + 1;
        for(int j = 0; j < patterns.getSize(); ++j)
            for(int i = 0; i < temp + 1; ++i)
            {
                B b(h.makeBuilder());
                for(int k = 0; k < q; ++k) b.add(patterns[j].first[i + k]);
                int hi = h(b);
                if(i == temp) candidates[hi].append(j);
                else shifts[hi] = min(temp - i, shifts[hi]);
            }
    }//return match position and next start position of (-1, -1) and indices
    //of patterns that match; allow different text and pattern types
    template<typename VECTOR2> pair<Vector<int>, int> findNext(
        VECTOR2 const& text, int textSize, int start = 0)
    {
        Vector<int> matches(patterns.getSize(), -1);
        while(start + minPatternSize <= textSize)
        {
            B b(h.makeBuilder());
            for(int j = 0; j < q; ++j)
                b.add(text[start + minPatternSize - q + j]);
            int hValue = h(b);
            bool foundAMatch = false;
            for(int i = 0; i < candidates[hValue].getSize(); ++i)
            {
                int j = candidates[hValue][i];
                if(start + patterns[j].second <= textSize && matchesAt(
                    start, text, patterns[j].first, patterns[j].second))
                {
                    foundAMatch = true;
                    matches[j] = start;
                }
            }
            start += shifts[hValue];
            if(foundAMatch) return make_pair(matches, start);
        }
        return make_pair(matches, -1);
    }
};

//Beware that the below code, though in principle correct, has a bug that I
//haven't had the time to fix. See the tests on the book's website for a
//failing use case
class RegularExpressionMatcher
{
    string re;
    int m;
    GraphAA<bool> g;

    Vector<int> findActiveStates(Vector<int> sources = Vector<int>())
    {
        Vector<bool> visited(g.nVertices(), false);
        DefaultDFSAction a;
        if(sources.getSize() == 0) DFSComponent(g, 0, visited, a);
        else for(int i = 0; i < sources.getSize(); ++i)
            if(!visited[sources[i]])
            {
                visited[sources[i]] = true;
                DFSComponent(g, sources[i], visited, a);
            }
        Vector<int> activeStates;
        for(int i = 0; i < visited.getSize(); ++i)
            if(visited[i]) activeStates.append(i);
        return activeStates;
    }
public:
    RegularExpressionMatcher(string const& theRe): re(theRe), m(re.length()),
        g(m + 1)
    {
        Stack<int> clauses;
        for(int i = 0; i < m; ++i)
        {
            int clauseStart = i;
            if(re[i] == '(' || re[i] == '|') clauses.push(i);
            else if(re[i] == ')')
            {
                int clauseOp = clauses.pop();
                if(re[clauseOp] == '|')
                {
                    clauseStart = clauses.pop();
                    g.addEdge(clauseStart, clauseOp+1);
                    g.addEdge(clauseOp, i);
                }
                else clauseStart = clauseOp;
            }
            if(i < m - 1 && re[i + 1]=='*')
                g.addUndirectedEdge(clauseStart, i + 1);
            if(re[i] == '(' || re[i] == '*' || re[i] == ')')
                g.addEdge(i, i + 1);
        }
    }

    bool matches(string const& text)
    {
        Vector<int> activeStates = findActiveStates();
        for(int i = 0; i < text.length(); ++i)
        {
            Vector<int> stillActive;
            for(int j = 0; j < activeStates.getSize(); ++j)
                if(activeStates[j] < m && re[activeStates[j]] == text[i])
                    stillActive.append(activeStates[j] + 1);
            activeStates = findActiveStates(stillActive);
        }
        for(int j = 0; j < activeStates.getSize(); ++j)
            if(activeStates[j] == m) return true;
        return false;
    }
};

class ShiftAndExtended
{//Joker handling omitted for simplicity
    enum{ALPHABET_SIZE = 1 << numeric_limits<unsigned char>::digits};
    unsigned char *pattern;
    int patternSize, position;//patternSize must be before masks
    unsigned long long charPos[ALPHABET_SIZE], O, P, L, R, state;
    unsigned long long makeMask(Vector<int> const& positions)const
    {
        unsigned long long mask = 0;
        for(int i = 0; i < positions.getSize(); ++i)
        {
            assert(positions[i] >= 0 && positions[i] < patternSize);
            Bits::set(mask, positions[i], true);
        }
        return mask;
    }
public:
    ShiftAndExtended(unsigned char* thePattern, int thePatternSize,
        Vector<int> const& repeatedPositions = Vector<int>(),
        Vector<int> const& optionalPositions = Vector<int>()): position(0),
        state(0), patternSize(thePatternSize), pattern(thePattern),
        R(makeMask(repeatedPositions)), O(makeMask(optionalPositions))
    {//first precompute character bit strings
        assert(patternSize <= numeric_limits<unsigned long long>::digits &&
            !Bits::get(O, 0));//position 0 can't be optional
        for(int i = 0; i < ALPHABET_SIZE; ++i) charPos[i] = 0;
        for(int i = 0; i < patternSize; ++i)
            Bits::set(charPos[pattern[i]], i, true);
        //then masks for optional characters
        unsigned long long sides = O ^ (O >> 1);
        P = (O >> 1) & sides;
        L = O & sides;
    }
    int findNext(unsigned char* text, int textSize)
    {
        while(position < textSize)
        {//first regular and repeatable update
            state = (((state << 1) | 1) | (state & R)) &
                charPos[text[position++]];
            //then optional character update
            unsigned long long sL = state | L;
            state |= O & (sL ^ ~(sL - P));
            if(Bits::get(state, patternSize - 1))return position - patternSize;
        }
        return -1;
    }
};

template<typename CHAR> class Diff
{
public:
    struct EditResult
    {
        CHAR c;
        int position;
        bool isInsert;
    };
private:
    struct Edit
    {
        Edit* prev;//used only for intermediate work and not the final result
        int position;
        bool isInsert;
    };
    static void extendDiagonal(int d, Vector<int>& frontierX, Vector<Edit*>&
        edits, Vector<CHAR> const& a, Vector<CHAR> const& b, Freelist<Edit>& f)
    {//pick next best edit
        int x = max(frontierX[d - 1] + 1, frontierX[d + 1]),
            y = x - (d - 1 - a.getSize());
        if(x != -1 || y != -1)
        {//apply it if not base case
            bool isInsert = x != frontierX[d + 1];
            edits[d] = new(f.allocate())Edit();
            edits[d]->isInsert = isInsert;
            edits[d]->prev = edits[d + (isInsert ? -1 : 1)];
            edits[d]->position = isInsert ? x : y;
        }//move diagonally as much as possible
        while(y + 1 < a.getSize() && x + 1 < b.getSize() &&
            a[y + 1] == b[x + 1])
        {
            ++y;
            ++x;
        }
        frontierX[d] = x;
    }
    static Vector<EditResult> DiffInternal(Vector<CHAR> const& a,
        Vector<CHAR> const& b, CHAR const& nullC)
    {
        int M = a.getSize(), N = b.getSize(), size = M + N + 3,
            mainDiagonal = N + 1;
        assert(M <= N);//a must be shorter then b for this helper
        Vector<int> frontierX(size, -2);
        Vector<Edit*> edits(size, 0);
        Freelist<Edit> f;
        int p = 0;
        for(; frontierX[mainDiagonal] < N - 1; ++p)
        {//from lower left to main
            for(int d = M + 1 - p; d < mainDiagonal; ++d)
                extendDiagonal(d, frontierX, edits, a, b, f);
            //from upper right to main
            for(int d = mainDiagonal + p; d >= mainDiagonal; --d)
                extendDiagonal(d, frontierX, edits, a, b, f);
        }//retrieve the computed path in reverse order
        Vector<EditResult> result;
        for(Edit* link = edits[mainDiagonal]; link; link = link->prev)
        {
            EditResult er = {nullC, link->position, link->isInsert};
            result.append(er);
        }//fix the order
        result.reverse();
        return result;
    }
public:
    static Vector<EditResult> diff(Vector<CHAR> const& a, Vector<CHAR> const&b,
        CHAR const& nullC = CHAR())//null char used for delete action as dummy
    {//edits needed to get a into b - positions are with respect to b
        bool exchange = a.getSize() > b.getSize();
        Vector<EditResult> result = exchange ? DiffInternal(b, a, nullC) :
            DiffInternal(a, b, nullC);
        for(int i = 0, netInserted = 0; i < result.getSize(); ++i)
        {//exchange if needed, set characters, and adjust deletion positions
            if(exchange) result[i].isInsert = !result[i].isInsert;
            if(result[i].isInsert)
            {
                ++netInserted;
                result[i].c = b[result[i].position];
            }
            else result[i].position += netInserted--;
        }
        return result;
    }
    static Vector<CHAR> applyDiff(Vector<CHAR> const& a,
        Vector<EditResult> const& script)
    {
        Vector<CHAR> b;
        int nextA = 0;
        for(int i = 0; i < script.getSize(); ++i)
        {//take chars from a until next edit position
            while(b.getSize() < script[i].position)
            {//basic input check - must not run out of a before next position
                assert(nextA < a.getSize());
                b.append(a[nextA++]);
            }
            if(script[i].isInsert) b.append(script[i].c);
            else ++nextA;//skip one a char on delete
        }//done with script, append the rest from a
        while(nextA < a.getSize()) b.append(a[nextA++]);
        return b;
    }
};

}//end namespace
#endif

