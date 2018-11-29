#ifndef COMPRESSION_H
#define COMPRESSION_H
#include "../RandomTreap/Trie.h"
#include "../Heaps/Heap.h"
#include "../StringAlgorithms/SuffixArray.h"
#include "Stream.h"
#include <cstdlib>
namespace igmdk{

Vector<unsigned char> ExtraBitsCompress(Bitset<unsigned char> const& bitset)
{
    assert(bitset.getSize() > 0);//makes no sense otherwise
    Vector<unsigned char> result = bitset.getStorage();
    result.append(bitset.garbageBits());
    return result;
}
Bitset<unsigned char> ExtraBitsUncompress(Vector<unsigned char> byteArray)
{
    assert(byteArray.getSize() > 1 && byteArray.lastItem() < BitStream::B);
    int garbageBits = byteArray.lastItem();
    byteArray.removeLast();
    Bitset<unsigned char> result(byteArray);
    while(garbageBits--) result.removeLast();
    return result;
}

void byteEncode(unsigned long long n, BitStream& result)
{
    enum{M05 = 1 << (numeric_limits<unsigned char>::digits - 1)};
    do
    {
        unsigned char r = n % M05;
        n /= M05;
        if(n) r += M05;
        result.writeByte(r);
    }while(n);
}
unsigned long long byteDecode(BitStream& stream)
{
    unsigned long long n = 0, base = 1;
    enum{M05 = 1 << (numeric_limits<unsigned char>::digits - 1)};
    for(;; base *= M05)
    {
        unsigned char code = stream.readByte(), value = code % M05;
        n += base * value;
        if(value == code) break;
    }
    return n;
}

void UnaryEncode(int n, BitStream& result)
{
    while(n--) result.writeBit(true);
    result.writeBit(false);
}
int UnaryDecode(BitStream& code)
{
    int n = 0;
    while(code.readBit()) ++n;
    return n;
}

void GammaEncode(unsigned long long n, BitStream& result)
{
    assert(n > 0);
    int N = lgFloor(n);
    UnaryEncode(N, result);
    if(N > 0) result.writeValue(n - twoPower(N), N);
}
unsigned long long GammaDecode(BitStream& code)
{
    int N = UnaryDecode(code);
    return twoPower(N) + (N > 0 ? code.readValue(N) : 0);
}

struct HuffmanTree
{
    enum{W = numeric_limits<unsigned char>::digits, N = 1 << W};
    struct Node
    {
        unsigned char letter;
        int count;
        Node *left, *right;
        Node(int theCount, Node* theLeft, Node* theRight,
             unsigned char theLetter): left(theLeft), right(theRight),
             count(theCount), letter(theLetter) {}
        bool operator<(Node const& rhs)const{return count < rhs.count;}
        void traverse(Bitset<unsigned char>* codebook,
            Bitset<unsigned char>& currentCode)
        {
            if(left)//internal node
            {
                currentCode.append(false);//went left
                left->traverse(codebook, currentCode);
                currentCode.removeLast();
                currentCode.append(true);//went right
                right->traverse(codebook, currentCode);
                currentCode.removeLast();
            }
            else codebook[letter] = currentCode;//leaf
        }
        void append(Bitset<unsigned char>& result)
        {
            result.append(!left);//0 for nonleaf, 1 for leaf
            if(left)
            {
                left->append(result);
                right->append(result);
            }
            else result.appendValue(letter, W);
        }
    }* root;
    Freelist<Node> f;

    HuffmanTree(Vector<unsigned char> const& byteArray)
    {//calculate frequencies
        int counts[N];
        for(int i = 0; i < N; ++i) counts[i] = 0;
        for(int i = 0; i < byteArray.getSize(); ++i) ++counts[byteArray[i]];
        //create leaf nodes
        Heap<Node*, PointerComparator<Node> > queue;
        for(int i = 0; i < N; ++i) if(counts[i] > 0) queue.insert(
            new(f.allocate())Node(counts[i], 0, 0, i));
        //merge leaf nodes to create the tree
        while(queue.getSize() > 1)//until forest merged
        {
            Node *first = queue.deleteMin(), *second = queue.deleteMin();
            queue.insert(new(f.allocate())
                Node(first->count + second->count, first, second, 0));
        }
        root = queue.getMin();
    }

    Node* readHuffmanTree(BitStream& text)
    {
        Node *left = 0, *right = 0;
        unsigned char letter;
        if(text.readBit()) letter = text.readValue(W);//got to a leaf
        else
        {//process internal nodes recursively
            left = readHuffmanTree(text);
            right = readHuffmanTree(text);
        }
        return new(f.allocate())Node(0, left, right, letter);
    }
    HuffmanTree(BitStream& text){root = readHuffmanTree(text);}

    void writeTree(Bitset<unsigned char>& result){root->append(result);}
    void populateCodebook(Bitset<unsigned char>* codebook)
    {
        Bitset<unsigned char> temp;
        root->traverse(codebook, temp);
    }

    Vector<unsigned char> decode(BitStream& text)
    //wrong bits will give wrong result, but not a crash
    {
        Vector<unsigned char> result;
        for(Node* current = root;;
            current = text.readBit() ? current->right : current->left)
        {
            if(!current->left)
            {
                result.append(current->letter);
                current = root;
            }
            if(!text.bitsLeft()) break;
        }
        return result;
    }
};

Vector<unsigned char> HuffmanCompress(Vector<unsigned char> const& byteArray)
{
    HuffmanTree tree(byteArray);
    Bitset<unsigned char> codebook[HuffmanTree::N], result;
    tree.populateCodebook(codebook);
    tree.writeTree(result);
    for(int i = 0; i < byteArray.getSize(); ++i)
        result.appendBitset(codebook[byteArray[i]]);
    return ExtraBitsCompress(result);
}

Vector<unsigned char> HuffmanUncompress(Vector<unsigned char> const& byteArray)
{
    BitStream text(ExtraBitsUncompress(byteArray));
    HuffmanTree tree(text);
    return tree.decode(text);
}

void LZWCompress(BitStream& in, BitStream& out, int maxBits = 16)
{
    assert(in.bytesLeft());
    byteEncode(maxBits, out);//store as config
    TernaryTreapTrie<int> dictionary;
    TernaryTreapTrie<int>::Handle h;
    int n = 0;
    while(n < (1 << numeric_limits<unsigned char>::digits))
    {//initialize with all bytes
        unsigned char letter = n;
        dictionary.insert(&letter, 1, n++);
    }
    Vector<unsigned char> word;
    while(in.bytesLeft())
    {
        unsigned char c = in.readByte();
        word.append(c);
        //if found keep appending
        if(!dictionary.findIncremental(word.getArray(), word.getSize(), h))
        {//word without the last byte guaranteed to be in the dictionary
            out.writeValue(*dictionary.find(word.getArray(),
                word.getSize() - 1), lgCeiling(n));
            if(n < twoPower(maxBits))//add new word if have space
                dictionary.insert(word.getArray(), word.getSize(), n++);
            word = Vector<unsigned char>(1, c);//set to read byte
        }
    }
    out.writeValue(*dictionary.find(word.getArray(), word.getSize()),
        lgCeiling(n));
}

void LZWUncompress(BitStream& in, BitStream& out)
{
    int maxBits = byteDecode(in), size = twoPower(maxBits), n = 0,
        lastIndex = -1;
    assert(maxBits >= numeric_limits<unsigned char>::digits);
    Vector<Vector<unsigned char> > dictionary(size);
    for(; n < (1 << numeric_limits<unsigned char>::digits); ++n)
        dictionary[n].append(n);
    while(in.bitsLeft())
    {
        int index = in.readValue(lastIndex == -1 ? 8 :
            min(maxBits, lgCeiling(n + 1)));
        if(lastIndex != -1 && n < size)
        {
            Vector<unsigned char> word = dictionary[lastIndex];
            word.append((index == n ? word : dictionary[index])[0]);
            dictionary[n++] = word;
        }
        for(int i = 0; i < dictionary[index].getSize(); ++i)
            out.writeByte(dictionary[index][i]);
        lastIndex = index;
    }
}

enum {RLE_E1 = (1 << numeric_limits<unsigned char>::digits) - 1,
    RLE_E2 = RLE_E1 - 1};
Vector<unsigned char> RLECompress(Vector<unsigned char>const& byteArray)
{
    Vector<unsigned char> result;
    for(int i = 0; i < byteArray.getSize();)
    {
        unsigned char byte = byteArray[i++];
        result.append(byte);
        int count = 0;
        while(count < RLE_E2 - 1 && i + count < byteArray.getSize() &&
            byteArray[i + count] == byte) ++count;
        if(count > 1 || (byte == RLE_E1 && count == 1))
        {
            result.append(RLE_E1);
            result.append(count);
            i += count;
        }
        else if(byte == RLE_E1) result.append(RLE_E2);
    }
    return result;
}
Vector<unsigned char> RLEUncompress(Vector<unsigned char>const& byteArray)
{
    Vector<unsigned char> result;
    for(int i = 0; i < byteArray.getSize();)
    {
        unsigned char byte = byteArray[i++];
        if(byte == RLE_E1 && byteArray[i] != RLE_E1)
        {
            unsigned char count = byteArray[i++];
            if(count == RLE_E2) count = 1;
            else byte = result.lastItem();//need temp if vector reallocates
            while(count--) result.append(byte);
        }
        else result.append(byte);
    }
    return result;
}

Vector<unsigned char> MoveToFrontTransform(bool compress,
    Vector<unsigned char>const& byteArray)
{
    unsigned char list[1 << numeric_limits<unsigned char>::digits], j, letter;
    for(int i = 0; i < sizeof(list); ++i) list[i] = i;
    Vector<unsigned char> resultArray;
    for(int i = 0; i < byteArray.getSize(); ++i)
    {
        if(compress)
        {//find and output rank
            j = 0;
            letter = byteArray[i];
            while(list[j] != letter) ++j;
            resultArray.append(j);
        }
        else
        {//rank to byte
            j = byteArray[i];
            letter = list[j];
            resultArray.append(letter);
        }//move list back to make space for front item
        for(; j > 0; --j) list[j] = list[j - 1];
        list[0] = letter;
    }
    return resultArray;
}

Vector<unsigned char> BurrowsWheelerTransform(
    Vector<unsigned char> const& byteArray)
{
    int original = 0, size = byteArray.getSize();
    Vector<int> BTWArray = suffixArray<BWTRank>(byteArray.getArray(), size);
    Vector<unsigned char> result;
    for(int i = 0; i < size; ++i)
    {
        int suffixIndex = BTWArray[i];
        if(suffixIndex == 0)
        {//found the original string
            original = i;
            suffixIndex = size;//avoid the % size in next step
        }
        result.append(byteArray[suffixIndex - 1]);
    }//assume that 4 bytes is enough
    Vector<unsigned char> code = ReinterpretEncode(original, 4);
    for(int i = 0; i < code.getSize(); ++i) result.append(code[i]);
    return result;
}

Vector<unsigned char> BurrowsWheelerReverseTransform(
     Vector<unsigned char> const& byteArray)
{
    enum{M = 1 << numeric_limits<unsigned char>::digits};
    int counts[M], firstPositions[M],
        textSize = byteArray.getSize() - 4;
    for(int i = 0; i < M; ++i) counts[i] = 0;
    Vector<int> ranks(textSize);//compute ranks
    for(int i = 0; i < textSize; ++i) ranks[i] = counts[byteArray[i]]++;
    firstPositions[0] = 0;//compute first positions
    for(int i = 0; i < M - 1; ++i)
        firstPositions[i + 1] = firstPositions[i] + counts[i];
    Vector<unsigned char> index, result(textSize);//extract original rotation
    for(int i = 0; i < 4; ++i) index.append(byteArray[i + textSize]);
    //construct in reverse order
    for(int i = textSize - 1, ix = ReinterpretDecode(index); i >= 0; --i)
        ix = ranks[ix] + firstPositions[result[i] = byteArray[ix]];
    return result;
}

Vector<unsigned char> BWTCompress(Vector<unsigned char>const& byteArray)
{
    return HuffmanCompress(RLECompress(MoveToFrontTransform(true,
        BurrowsWheelerTransform(byteArray))));
}
Vector<unsigned char> BWTUncompress(Vector<unsigned char>const& byteArray)
{
    return BurrowsWheelerReverseTransform(MoveToFrontTransform(false,
       RLEUncompress(HuffmanUncompress(byteArray))));
}

}
#endif
