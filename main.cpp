#include "utils.hpp"
#include "NeuralNetwork.hpp"
#include <iostream>
#include <array>    // for storing weights
#include <random>   // for initializing weights
#include <map>      // for min/max identification
using namespace std;

const int N_ATTRS = 7;
const int N_CLASSES = 3;

// parse attributes
classification_dataset prep_dataset(parsed_file& file,bool verbose = false) {
  /*
  columns 0-6 = attributes = float
  columns 7   = class      = int
  */
  // containers to identify columnwise mins and maxs
  bool initialize_min_max = true;             // logic flag re "should initialize min max
  observation_attributes attr_min(N_ATTRS,0); //{numeric_limits<double>::max()};
  observation_attributes attr_max(N_ATTRS,0); //{numeric_limits<double>::min()};

  // map classes to integer
  map<string,int> class_mapper;
  int class_counter = 0; // counter for unique classes

  // container to store massaged data
  classification_dataset dataset;

  // parse + massage data
  for (auto it = file.begin();it!=file.end();it++) {
    word_vector vw = *it;
    // containers for storing useable data
    observation_attributes attrs(N_ATTRS,0); // convert attributes from string to doubles
    observation_class mapped_class;          // convert class from string to int

    // convert string attrs to floats
    for (size_t i=0;i<attrs.size();i++)
      attrs[i] = stod(vw[i]);

    // identify mins/maxs for each column
    if (initialize_min_max) {
      // ensure attr_min and attr_max populated sensibly
      for (size_t i=0;i<attrs.size();i++) {
        attr_min[i] = attrs[i];
        attr_max[i] = attrs[i];
      }
      initialize_min_max = false;
    } else {
      for (size_t i=0;i<attrs.size();i++) {
        if (attrs[i]<attr_min[i]) attr_min[i] = attrs[i];
        if (attrs[i]>attr_max[i]) attr_max[i] = attrs[i];
      }
    }

    // map all classes to 0,...,N
    string class_string = vw[N_ATTRS];
    if (class_mapper.find(class_string) == class_mapper.end()) {
      // class not present
      class_mapper[class_string] = class_counter; // give new class a name
      class_counter++;                             // increment for next unique class
      /*
      // show class mappings
      for (auto it = class_mapper.begin();it!=class_mapper.end();it++) cout << it->first << " : "<< it->second << endl;
      cout << class_counter << endl;
      */

    }
    mapped_class = class_mapper[class_string];
    //cout << "raw = " << class_string << " mapped = " << mapped_class << endl;

    classification_observation observation;
    observation.first = mapped_class;
    observation.second = attrs;

    // save parsed dataset
    dataset.push_back(observation);
  }

  // normalized attrs to [0-1] with min-max normalizer
  for (auto it = dataset.begin();it!=dataset.end();it++) {
    observation_attributes& attrs = it->second;
    for (size_t i=0;i<attrs.size();i++)
      attrs[i] = (attrs[i] - attr_min[i]) / (attr_max[i] - attr_min[i]);
  }

  // show workings
  if (verbose) {
    // show min/max on each column
    cout << "attribute mins / maxs" << endl;
    cout << "\t";
    for (size_t i=0;i<attr_min.size();i++) cout << attr_min[i] << " ";
    cout << endl;
    cout << "\t";
    for (size_t i=0;i<attr_max.size();i++) cout << attr_max[i] << " ";
    cout << endl;
    cout << endl;

    // show class mappings
    cout << "class mappings" << endl;
    for (auto it = class_mapper.begin();it!=class_mapper.end();it++) cout << "\t" << it->first << " : "<< it->second << endl;
    cout << endl;

    // display normalized dataset
    for (auto observation:dataset) {
      cout << observation.first << " : "
      << observation.second[0] << " "
      << observation.second[1] << " "
      << observation.second[2] << " "
      << observation.second[3] << " "
      << observation.second[4] << " "
      << observation.second[5] << " "
      << observation.second[6] << " "
      << endl;
    }
  }

  // return normalized dataset
  return dataset;
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

  // get data
  parsed_file file = parse_csv("wheat-seeds.txt");          // parse
  classification_dataset dataset = prep_dataset(file,true); // convert data to numeric, normalize and categorize classes

  // training params
  int n_folds = 5;
  double learning_rate = 0.05;
  int n_epochs = 1000000;
  int n_hidden = 20;
  bool verbose = true;
  int verbose_n_epochs = 100;

  NeuralNetwork nn = NeuralNetwork(N_ATTRS,n_hidden,N_CLASSES);
  nn.train_n_epochs(dataset,learning_rate,n_epochs,verbose,verbose_n_epochs);

  return 0;
}
