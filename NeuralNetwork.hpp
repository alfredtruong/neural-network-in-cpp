#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Node.hpp"
#include "Layer.hpp"
#include <vector>

using namespace std;

////////////////////////////////////////////////////////
// TYPES
////////////////////////////////////////////////////////

typedef vector<double> observation_attributes;                                     // observation attributes
typedef int observation_class;                                                     // observation class
typedef pair<observation_class,observation_attributes> classification_observation; // classification observation
typedef vector<classification_observation> classification_dataset;                 // vector of observations for classification

////////////////////////////////////////////////////////
// NEURAL NETWORK
////////////////////////////////////////////////////////

class NeuralNetwork
{
private:
  int m_nLayers;           // number of layers
  int m_nInputNodes;       // size of input layer
  int m_nHiddenNodes;      // size of hidden layer
  int m_nOutputNodes;      // size of output layer
  vector<Layer> m_layers;

public:
  // constructor
  NeuralNetwork(int n_inputs,int n_hiddens,int n_outputs);   // constructor
  ~NeuralNetwork();                                          // destructor

  // methods
  void add_layer(string name,int n_inputs,int n_nodes);
  void forward_propagate_input(const vector<double>& input);
  void back_propagate_errors(const vector<double>& expected_output);
  void update_weights(const vector<double>& input, double learning_rate);

  void train_single_observation(const classification_observation& observation, double learning_rate);
  void train_entire_dataset(const classification_dataset& dataset,double learning_rate);
  void train_n_epochs(const classification_dataset& dataset,double learning_rate, int n_epochs,bool verbose,int verbose_epochs);
  void evaluate_network(const classification_dataset& dataset,int epoch);
  vector<double> predict(const vector<double>& input);
  const vector<double> classID_to_onehot(observation_class classID);
  double output_squared_error(const vector<double>& output,const vector<double>& expected_output);

  // print
  void print(void);

};

void NeuralNetwork_test_instantiation(void);
void NeuralNetwork_test_forward_propagation(void);

#endif
