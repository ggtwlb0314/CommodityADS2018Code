#ifndef GC_FREE_LIST_H
#define GC_FREE_LIST_H
#include "Utils.h"
#include "Vector.h"
namespace igmdk{

template<typename ITEM> class SimpleDoublyLinkedList
{
    struct Node
    {
        ITEM item;
        Node *next, *prev;
        template<typename ARGUMENT>
        Node(ARGUMENT const& a): item(a), next(0), prev(0) {}
    } *root, *last;
    void cut(Node* n)
    {//join prev and next
        assert(n);
        (n == last ? last : n->next->prev) = n->prev;
        (n == root ? root : n->prev->next) = n->next;
    }
public:
    SimpleDoublyLinkedList(): root(0), last(0){}
    template<typename ARGUMENT> append(ARGUMENT const& a)
    {
        Node* n = new Node(a);
        n->prev = last;
        if(last) last->next = n;
        last = n;
        if(!root) root = n;
    }
    class Iterator
    {
        Node* current;
    public:
        Iterator(Node* n): current(n){}
        typedef Node* Handle;
        Handle getHandle(){return current;}
        Iterator& operator++()
        {
            assert(current);
            current = current->next;
            return *this;
        };
        Iterator& operator--()
        {
            assert(current);
            current = current->prev;
            return *this;
        };
        ITEM& operator*()const{assert(current); return current->item;}
        ITEM* operator->()const{assert(current); return &current->item;}
        bool operator==(Iterator const& rhs)const
            {return current == rhs.current;}
    };
    Iterator begin(){return Iterator(root);}
    Iterator rBegin(){return Iterator(last);}
    Iterator end(){return Iterator(0);}
    Iterator rEnd(){return Iterator(0);}
    void moveBefore(Iterator what, Iterator where)
    {//where is allowed to be 0 which means move to end
        assert(what != end());
        if(what != where)
        {
            Node *n = what.getHandle(), *w = where.getHandle();
            cut(n);
            n->next = w;
            if(w)
            {
                n->prev = w->prev;
                w->prev = n;
            }
            else
            {
                n->prev = last;
                last = n;
            }
            if(n->prev) n->prev->next = n;
            if(w == root) root = n;
        }
    }
    template<typename ARGUMENT> prepend(ARGUMENT const& a)
    {
        append(a);
        moveBefore(rBegin(), begin());
    }
    void remove(Iterator what)
    {
        assert(what != end());
        cut(what.getHandle());
        delete what.getHandle();
    }
    SimpleDoublyLinkedList(SimpleDoublyLinkedList const& rhs)
        {for(Node* n = rhs.root; n; n = n->next){append(n->item);}}
    SimpleDoublyLinkedList& operator=(SimpleDoublyLinkedList const&rhs)
        {return genericAssign(*this,rhs);}
    ~SimpleDoublyLinkedList()
    {
        while(root)
        {
            Node* toBeDeleted = root;
            root = root->next;
            delete toBeDeleted;
        }
    }
};

template<typename ITEM> struct StaticFreelist
{
    int capacity, size, maxSize;
    struct Item
    {
        ITEM item;
        union
        {
            Item* next;
            void* userData;
        };
    } *nodes, *returned;
    StaticFreelist(int fixedSize): capacity(fixedSize), size(0), maxSize(0),
        returned(0), nodes(rawMemory<Item>(fixedSize)){}
    bool isFull(){return size == capacity;}
    bool isEmpty(){return size <= 0;}
    Item* allocate()
    {
        assert(!isFull());
        Item* result = returned;
        if(result) returned = returned->next;
        else result = &nodes[maxSize++];
        ++size;
        return result;
    }
    void remove(Item* item)
    {//nodes must come from this list
        assert(item - nodes >= 0 && item - nodes < maxSize);
        item->item.~ITEM();
        item->next = returned;
        returned = item;
        --size;
    }
    ~StaticFreelist()
    {
        if(!isEmpty())
        {//mark allocated nodes, unmark returned ones, destruct marked ones
            Vector<bool> toDestruct(maxSize, true);
            while(returned)
            {
                toDestruct[returned - nodes] = false;
                returned = returned->next;
            }
            for(int i = 0; i < maxSize; ++i)
                if(toDestruct[i])nodes[i].item.~ITEM();
        }
        rawDelete(nodes);
    }
};

template<typename ITEM> class Freelist
{
    enum{MAX_BLOCK_SIZE = 8192, MIN_BLOCK_SIZE = 8, DEFAULT_SIZE = 32};
    int blockSize;
    typedef SimpleDoublyLinkedList<StaticFreelist<ITEM> > ListType;
    typedef typename StaticFreelist<ITEM>::Item Item;
    typedef typename ListType::Iterator I;
    ListType blocks;
    //disallow copying
    Freelist(Freelist const&);
    Freelist& operator=(Freelist const&);
public:
    Freelist(int initialSize = DEFAULT_SIZE): blockSize(max<int>(
        MIN_BLOCK_SIZE, min<int>(initialSize, MAX_BLOCK_SIZE))) {}
    ITEM* allocate()
    {
        I first = blocks.begin();
        if(first == blocks.end() || first->isFull())
        {//make new first node if needed
            blocks.prepend(blockSize);
            first = blocks.begin();
            blockSize = min<int>(blockSize * 2, MAX_BLOCK_SIZE);
        }
        Item* result = first->allocate();
        result->userData = (void*)first.getHandle();
        //move full blocks to the end
        if(first->isFull()) blocks.moveBefore(first, blocks.end());
        return (ITEM*)result;
    }
    void remove(ITEM* item)
    {
        if(!item) return;//handle null pointer
        Item* node = (Item*)(item);
        I cameFrom((typename I::Handle)node->userData);
        cameFrom->remove(node);
        if(cameFrom->isEmpty())
        {//delete block if empty, else reduce its size
         //beware that block boundary delete/remove thrashes, but unlikely
            blockSize = max<int>(MIN_BLOCK_SIZE,
                blockSize - cameFrom->capacity);
            blocks.remove(cameFrom);
        }//move available blocks to the front
        else blocks.moveBefore(cameFrom, blocks.begin());
    }
};

}//end namespace
#endif
