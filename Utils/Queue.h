#ifndef QUEUE_H
#define QUEUE_H
#include "Utils.h"
namespace igmdk{

template<typename ITEM> class Queue
{
    enum{MIN_CAPACITY = 8};
    int capacity, front, size;
    ITEM* items;
    int offset(int i)const{return (front + i) % capacity;}
    void resize()
    {
        ITEM* oldArray = items;
        int newCapacity = max(int(MIN_CAPACITY), size * 2);
        items = rawMemory<ITEM>(newCapacity);
        for(int i = 0; i < size; ++i)new(&items[i])ITEM(oldArray[offset(i)]);
        deleteArray(oldArray);
        front = 0;
        capacity = newCapacity;
    }
    void deleteArray(ITEM* array)
    {
        for(int i = 0; i < size; ++i) array[offset(i)].~ITEM();
        rawDelete(array);
    }
public:
    bool isEmpty()const{return size == 0;}
    int getSize()const{return size;}
    ITEM& operator[](int i)
    {
        assert(i >= 0 && i < size);
        return items[offset(i)];
    }
    ITEM const& operator[](int i)const
    {
        assert(i >= 0 && i < size);
        return items[offset(i)];
    }
    Queue(int theSize = MIN_CAPACITY): capacity(max(int(MIN_CAPACITY),
        theSize)), front(0), size(0), items(rawMemory<ITEM>(capacity)) {}
    Queue(Queue const& rhs): capacity(max(int(MIN_CAPACITY), rhs.size)),
        size(rhs.size), front(0), items(rawMemory<ITEM>(capacity))
        {for(int i = 0; i < size; ++i) push(rhs[i]);}
    Queue& operator=(Queue const& rhs){return genericAssign(*this, rhs);}
    ~Queue(){deleteArray(items);}
    void push(ITEM const& item)
    {
        if(size == capacity) resize();
        new(&items[offset(size++)])ITEM(item);
    }
    ITEM pop()
    {
        assert(!isEmpty());
        ITEM result = items[front];
        items[front].~ITEM();
        front = offset(1);
        if(capacity > 4 * --size && capacity > MIN_CAPACITY) resize();
        return result;
    }
    ITEM& top()const
    {
        assert(!isEmpty());
        return items[front];
    }
};
}
#endif
