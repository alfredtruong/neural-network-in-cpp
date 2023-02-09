#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Node.hpp"
#include "Layer.hpp"
#include <vector>

using namespace std;

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
  void forward_propagate_inputs(const vector<double>& inputs);
  void back_propagate_errors(const vector<double>& expected_outputs);
  void update_weights(const vector<double>& inputs, double learning_rate);
  void train();
  vector<double> predict(const vector<double>& inputs);

  // print
  void print(void);

};

void NeuralNetwork_test_instantiation(void);
void NeuralNetwork_test_forward_propagation(void);

#endif
