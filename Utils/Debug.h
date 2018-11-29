#ifndef DEBUG_H
#define DEBUG_H
#include <iostream>
#include <iomanip>
using namespace std;
namespace igmdk{

#define DEBUG(var) cout << #var " "<< setprecision(17) << (var) << endl;

}//end namespace
#endif
