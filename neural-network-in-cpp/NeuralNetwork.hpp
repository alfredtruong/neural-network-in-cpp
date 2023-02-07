#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.hpp"

using namespace std;

class NeuralNetwork
{
private:
  int m_nInputs;
  int m_nHiddens;
  int m_nOutputs;
  vector<Layer> m_layers;

public:
  // constructor
  NeuralNetwork(int n_inputs,int n_hiddens,int n_outputs);
  ~NeuralNetwork();

  // methods
  void forward_propagate_inputs(vector<double>& inputs);
  void back_propagate_errors(vector<double>& expected_outputs);
  void update_weights();

  // print
  void print(void);
};

#endif
