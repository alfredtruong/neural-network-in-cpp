#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

////////////////////////////////////////////////////////
// TEMPLATES
////////////////////////////////////////////////////////

template <typename T>
void display_vector(vector<T>& vec,const string name) {
  // show outputs
  cout << "\t[" << name << "] n = [" << vec.size() << "], data = [ ";
  for (auto x: vec) cout << x << ", ";
  cout << "]" << endl;
}

template <typename T>
int argmax(vector<T> vec) {
  return distance(vec.begin(),max_element(vec.begin(),vec.end()));
}

template <typename T>
int argmin(vector<T> vec) {
  return distance(vec.begin(),min_element(vec.begin(),vec.end()));
}

////////////////////////////////////////////////////////
// UTILS
////////////////////////////////////////////////////////
typedef string word;
typedef vector<word> word_vector;
typedef vector<word_vector> parsed_file;
parsed_file parse_csv(string filename,bool verbose = false);

#endif
