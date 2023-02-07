#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>

using namespace std;

////////////////////////////////////////////////////////
// UTILS
////////////////////////////////////////////////////////
template <typename T>
void display_vector(vector<T>& v,const string name) {
  // show outputs
  cout << "\t[" << name << "] n = [" << v.size() << "], data = [ ";
  for (auto x: v) cout << x << ", ";
  cout << "]" << endl;
}

#endif
