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

typedef int obs_class;
typedef array<double,N_ATTRS> obs_attrs;
typedef pair<obs_class,obs_attrs> y_X;
typedef vector<y_X> dataset;

typedef vector<vector<string>> parsed_file;

// parse attributes
dataset prep_dataset(parsed_file& file) {
  /*
  columns 0-6 = attributes = float
  columns 7   = classe     = int
  */
  // containers to identify columnwise mins and maxs
  bool initialize_min_max = true; // logic flag re "should initialize min max
  obs_attrs attr_min; //{numeric_limits<double>::max()};
  obs_attrs attr_max; //{numeric_limits<double>::min()};

  // map classes to integer
  map<string,int> class_mapper;
  int class_counter = 0; // counter for unique classes

  // container to store massaged data
  dataset ds;

  // parse + massage data
  for (auto it = file.begin();it!=file.end();it++) {
    vector<string> row = *it;
    // convert string to useable data
    obs_attrs attrs;        // convert attributes from string to doubles
    obs_class mapped_class; // convert class from string to int


    for (int i=0;i<N_ATTRS;i++)
      attrs[i] = stod(row[i]);

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
      class_mapper[row[N_ATTRS]] == class_counter;
      class_counter++;
    }

    // save parsed dataset
    ds.push_back();
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
