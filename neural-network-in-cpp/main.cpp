#include "utils.hpp"
#include "Node.hpp"
#include "Layer.hpp"
#include "NeuralNetwork.hpp"
#include <iostream>
#include <array>    // for storing weights
#include <random>   // for initializing weights
#include <map>      // for min/max identification
using namespace std;

const int N_ATTRS = 7;

typedef array<double,N_ATTRS> obs_attrs;
typedef int obs_class;
typedef pair<obs_attrs,obs_class> X_y;
typedef vector<X_y> dataset;

X_y parsed_line_to_X_y(vector<string>& parsed_line) {
  /*
  0-6 = Xs = float
  7   = y  = int
  */
  // parse attributes
  obs_attrs row_attrs;
  for (int i=0;i<N_ATTRS;i++)
    row_attrs[i] = stod(parsed_line[i]);

  // parse class
  obs_class row_class = stoi(parsed_line[7]);

  return X_y(row_attrs,row_class);
}

dataset parsed_file_to_dataset(vector<vector<string>> parsed_file) {
  // containers to identify columnwise mins and maxs
  bool initialize_min_max = true; // logic flag re "should initialize min max
  obs_attrs attr_min; //{numeric_limits<double>::max()};
  obs_attrs attr_max; //{numeric_limits<double>::min()};

  // dico to map classes to integers
  map<string,int> class_mapper;
  int class_counter = 0; // counter for unique classes

  // container to store massaged data
  dataset ds;

  // parse + massage data
  for (auto it = parsed_file.begin();it!=parsed_file.end();it++) {
    // convert attributes from strong to doubles
    obs_attrs attrs;
    obs_class mapped_class;

    for (int i=0;i<N_ATTRS;i++)
      attrs[i] = stod((*it)[i]);

    // identify min/max on each column
    if (initialize_min_max) {
      // ensure attr_min and attr_max populated sensibly
      for (int i=0;i<N_ATTRS;i++) {
        attr_min[i] = attrs[i];
        attr_max[i] = attrs[i];
      }
      initialize_min_max = false;
    } else {
      for (int i=0;i<N_ATTRS;i++) {
        if (attrs[i]<attr_min[i]) attr_min[i] = attrs[i];
        if (attrs[i]>attr_max[i]) attr_max[i] = attrs[i];
      }
    }

    // map classes to 0,...,N
    if (class_mapper.find((*it)[N_ATTRS]) == class_mapper.end()) {
      // class not present
      class_mapper[(*it)[N_ATTRS]] == class_counter;
      class_counter++;
    }

    // save parsed dataset
    ds.push_back(X_y(attrs,));
  }

  // show min/max on each column
  for (int i=0;i<N_ATTRS;i++) cout << attr_min[i] << " ";
  cout << endl;
  for (int i=0;i<N_ATTRS;i++) cout << attr_max[i] << " ";
  cout << endl;

  // show class mappings
  for (auto it = class_mapper.begin();it!=class_mapper.end();it++)
    cout << it->first << it->second << endl;

  // normalize attributes and categorize classes
  for (auto it = ds.begin();it!=ds.end();it++) {
    // normalized ds
    for (int i=0;i<N_ATTRS;i++) {
      attrs[i] = (attrs[i] - attr_min[i]) / (attr_max[i] - attr_min[i]);
    }

    // categorize classes
    mapped_class = class_mapper[(*it)[N_ATTRS];
  }

  // return normalized ds
  return X_y(attrs,mapped_class);
}

int main() {

  //Node_test_instantiation(); // instantiation
  //Node_test_forward_propagation(); // forward propagation

  //Layer_test_instantiation(); // instantiation
  //Layer_test_forward_propagation(); // forward propagation

  //NeuralNetwork_test_instantiation(); // instantiation
  //NeuralNetwork_test_forward_propagation(); // forward propagation


  // test instantiation
  // read input data

  // train it

  vector<vector<string>> parsed_file = parse_csv("wheat-seeds.txt");
  dataset ds = parsed_file_to_dataset(parsed_file);
  for (auto row:ds) {
    cout << row.second << " : "
    << row.first[0] << " "
    << row.first[1] << " "
    << row.first[2] << " "
    << row.first[3] << " "
    << row.first[4] << " "
    << row.first[5] << " "
    << row.first[6] << " "
    << endl;
  }
  return 0;
}
