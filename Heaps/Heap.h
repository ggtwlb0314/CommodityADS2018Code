#ifndef PRIORITY_QUEUE_H
#define PRIORITY_QUEUE_H
#include "../Utils/Vector.h"
#include "../HashTable/ChainingHashTable.h"
namespace igmdk{

template<typename ITEM>
struct ReportDefault{void operator()(ITEM& item, int i){}};
template<typename ITEM, typename COMPARATOR = DefaultComparator<ITEM>,
    typename REPORTER = ReportDefault<ITEM> > class Heap
{
    REPORTER r;
    int getParent(int i)const{return (i - 1)/2;}
    int getLeftChild(int i)const{return 2 * i + 1;}
    Vector<ITEM> items;
    void moveUp(int i)
    {
        ITEM temp = items[i];
        for(int parent; i > 0 && c(temp, items[parent =
             getParent(i)]); i = parent) r(items[i] = items[parent], i);
        r(items[i] = temp, i);
    }
    void moveDown(int i)
    {
        ITEM temp = items[i];
        for(int child; (child = getLeftChild(i)) < items.getSize();
            i = child)
        {//find smaller child
            int rightChild = child + 1;
            if(rightChild < items.getSize() && c(items
                [rightChild], items[child])) child = rightChild;
            //replace with the smaller child if any
            if(!c(items[child], temp)) break;
            r(items[i] = items[child], i);
        }
        r(items[i] = temp, i);
    }
public:
    COMPARATOR c;
    Heap(COMPARATOR const& theC = COMPARATOR(), REPORTER const&
        theReporter = REPORTER()): r(theReporter), c(theC) {}
    bool isEmpty()const{return items.getSize() == 0;}
    int getSize()const{return items.getSize();}
    ITEM const& getMin()const
    {
        assert(!isEmpty());
        return items[0];
    }
    void insert(ITEM const& item)
    {
        items.append(item);
        moveUp(items.getSize() - 1);
    }
    ITEM const& operator[](int i)const
    {
        assert(i >= 0 && i < items.getSize());
        return items[i];
    }
    void changeKey(int i, ITEM const& item)
    {
        assert(i >= 0 && i < items.getSize());
        bool decrease = c(item, items[i]);
        items[i] = item;
        decrease ? moveUp(i) : moveDown(i);
    }
    ITEM deleteMin(){return remove(0);}
    ITEM remove(int i)
    {
        assert(i >= 0 && i < items.getSize());
        ITEM result = items[i];
        r(result, -1);
        if(items.getSize() > i)
        {//not last item
            items[i] = items.lastItem();
            r(items[i], i);//report move
            moveDown(i);//won't touch the last item
        }
        items.removeLast();
        return result;
    }
};

template<typename ITEM, typename COMPARATOR = DefaultComparator<ITEM>,
    typename HANDLE = int, typename HASHER = EHash<BUHash> > class IndexedHeap
{
    ChainingHashTable<HANDLE, int> map;
    typedef typename ChainingHashTable<HANDLE, int>::NodeType* POINTER;
    typedef pair<ITEM, POINTER> Item;
    typedef PairFirstComparator<ITEM, POINTER, COMPARATOR> Comparator;
    struct Reporter
        {void operator()(Item& item, int i){item.second->value = i;}};
    Heap<Item, Comparator, Reporter> h;
public:
    IndexedHeap(COMPARATOR const& theC = COMPARATOR()): h(Comparator(theC)) {}
    int getSize()const{return h.getSize();}
    ITEM const* find(HANDLE handle)
    {
        int* index = map.find(handle);
        return index ? &h[*index].first : 0;
    }
    bool isEmpty()const{return h.isEmpty();}
    void insert(ITEM const& item, HANDLE handle)
        {h.insert(Item(item, map.insert(handle, h.getSize())));}
    pair<ITEM, HANDLE> getMin()const
    {
        Item temp = h.getMin();
        return make_pair(temp.first, temp.second->key);
    }
    pair<ITEM, HANDLE> deleteMin()
    {
        Item temp = h.deleteMin();
        pair<ITEM, HANDLE> result = make_pair(temp.first, temp.second->key);
        map.remove(temp.second->key);
        return result;
    }
    void changeKey(ITEM const& item, HANDLE handle)
    {
        POINTER p = map.findNode(handle);
        if(p) h.changeKey(p->value, Item(item, p));
        else insert(item, handle);
    }
    void deleteKey(HANDLE handle)
    {
        int* index = map.find(handle);
        assert(index);
        h.remove(*index);
        map.remove(handle);
    }
};

template<typename ITEM, typename COMPARATOR = DefaultComparator<ITEM> >
class IndexedArrayHeap
{
    Vector<int> map;
    typedef pair<ITEM, int> Item;
    typedef PairFirstComparator<ITEM, int, COMPARATOR> Comparator;
    struct Reporter
    {
        Vector<int>& pmap;
        Reporter(Vector<int>& theMap): pmap(theMap) {}
        void operator()(Item& item, int i){pmap[item.second] = i;}
    };
    Heap<Item, Comparator, Reporter> h;
public:
    typedef Item ITEM_TYPE;
    IndexedArrayHeap(COMPARATOR const& theC = COMPARATOR()):
        h(Comparator(theC), Reporter(map)) {}
    int getSize()const{return h.getSize();}
    ITEM const* find(int handle)
    {
        int pointer = map[handle];
        return pointer != -1 ? &h[pointer].first : 0;
    }
    bool isEmpty()const{return h.isEmpty();}
    void insert(ITEM const& item, int handle)
    {
        if(handle >= map.getSize())
            for(int i = map.getSize(); i <= handle; ++i) map.append(-1);
        h.insert(Item(item, handle));
    }
    pair<ITEM, int> const& getMin()const{return h.getMin();}
    pair<ITEM, int> deleteMin()
    {
        Item result = h.deleteMin();
        map[result.second] = -1;
        return result;
    }
    void changeKey(ITEM const& item, int handle)
    {
        int p = map[handle];
        if(p != -1) h.changeKey(p, Item(item, handle));
        else insert(item, handle);
    }
    void deleteKey(int handle)
    {
        int pointer = map[handle];
        assert(pointer != -1);
        h.remove(pointer);
        map[handle] = -1;
    }
};

}
#endif
