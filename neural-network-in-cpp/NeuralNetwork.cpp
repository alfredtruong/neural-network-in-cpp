#include "NeuralNetwork.hpp"
#include "utils.hpp"
#include <iostream>

////////////////////////////////////////////////////////
// NETWORK
////////////////////////////////////////////////////////

// initialize network
NeuralNetwork::NeuralNetwork(int n_inputs,int n_hiddens,int n_outputs) {
  m_nInputs = n_inputs;
  m_nHiddens = n_hiddens;
  m_nOutputs = n_outputs;

  m_nLayers = 0;
  add_layer("hidden",n_inputs,n_hiddens);
  add_layer("output",n_hiddens,n_outputs);
};

// destructor
NeuralNetwork::~NeuralNetwork() {
};

// add layer
void NeuralNetwork::add_layer(string name,int n_inputs,int n_nodes) {
  m_layers.push_back(Layer(name,n_inputs,n_nodes));
  m_nLayers++;
};

// forward iterate over layers
void NeuralNetwork::forward_propagate_inputs(const vector<double>& inputs) {
  vector<double> new_inputs = inputs;
  for (int i=0;i<m_nLayers;i++) {
    Layer& this_layer = m_layers[i];
    this_layer.evaluate_inputs(new_inputs);
    new_inputs = this_layer.get_outputs(); // next layer looks at this layers outputs
  }
}

// reverse iterate over layers
void NeuralNetwork::back_propagate_errors(const vector<double>& expected_outputs) {
  for (int i=m_nLayers-1;i>=0;i--) {
    Layer& this_layer = m_layers[i];
    if (int i=m_nLayers-1) {
      this_layer.update_output_layer_deltas(expected_outputs); // output layer error = "output - expected"
    } else {
      Layer& next_layer = m_layers[i+1];
      this_layer.update_hidden_layer_deltas(next_layer);       // hidden layer error = "weighted error of next level projected back"
    }
  }
}

// forward iterate over layers
void NeuralNetwork::update_weights(const vector<double>& inputs, double learning_rate) {
  vector<double> new_inputs = inputs;
  for (int i=0;i<m_nLayers;i++) {
    Layer& this_layer = m_layers[i];
    this_layer.update_weights(new_inputs,learning_rate);
    new_inputs = this_layer.get_outputs(); // next layer looks at this layers outputs
  }
}

void NeuralNetwork::print(void) {
  cout << "network [" << this << "]";
  cout << ", nInputs = [" << m_nInputs << "]";
  cout << ", nHiddens = [" << m_nHiddens << "]";
  cout << ", nOutputs = [" << m_nOutputs << "]";
  cout << endl;
  for (auto layer : m_layers) layer.print();
  cout << endl;
}

vector<double> NeuralNetwork::predict(const vector<double>& inputs) {
  forward_propagate_inputs(inputs);
  Layer& output_layer = m_layers[m_nLayers-1];
  return output_layer.get_outputs();
}

void NeuralNetwork_test_instantiation(void) {
  cout << __func__ << endl << endl;

  NeuralNetwork nn1 = NeuralNetwork(3,2,4);
  NeuralNetwork nn2 = NeuralNetwork(2,2,5);
  nn1.print();
  nn2.print();
}

void NeuralNetwork_test_forward_propagation(void) {
  cout << __func__ << endl << endl;

  // instantiation
  NeuralNetwork nn1 = NeuralNetwork(3,2,4);
  nn1.print();

  // forward propagation
  vector<double> inputs = {1,2,3};
  nn1.forward_propagate_inputs(inputs);
  nn1.print();

  // back propagate errors
  vector<double> expected_outputs = {3,4,5,6};
  nn1.back_propagate_errors(expected_outputs);
  nn1.print();

  // update weights
  nn1.update_weights(inputs,0.5);
  nn1.print();
}
