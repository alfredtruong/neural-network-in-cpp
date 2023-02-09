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
  int m_nLayers;           // counter of layers
  int m_nInputs;           // counter of inputs
  int m_nHiddens;          // counter of hidden nodes
  int m_nOutputs;          // counter of outputs
  vector<Layer> m_layers;

public:
  // constructor
  NeuralNetwork(int n_inputs,int n_hiddens,int n_outputs);   // constructor
  ~NeuralNetwork();                                          // destructor

  // methods
  void add_layer(string name,int n_inputs,int n_nodes);
  void forward_propagate_input(const vector<double>& input);
  void back_propagate_errors(const vector<double>& expected_outputs);
  void update_weights(const vector<double>& input, double learning_rate);
  vector<double> predict(const vector<double>& input);
  int outputs_to_classID(const vector<double>& outputs);
  void train(classification_dataset dataset, int nClasses, double learning_rate, int n_epoch);

  // print
  void print(void);

};

void NeuralNetwork_test_instantiation(void);
void NeuralNetwork_test_forward_propagation(void);

#endif
