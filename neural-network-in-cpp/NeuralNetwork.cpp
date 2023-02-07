#include "utils.hpp"
#include "NeuralNetwork.hpp"

////////////////////////////////////////////////////////
// NETWORK
////////////////////////////////////////////////////////

// initialize network
NeuralNetwork::NeuralNetwork(int n_inputs,int n_hiddens,int n_outputs) {
  m_nInputs = n_inputs;
  m_nHiddens = n_hiddens;
  m_nOutputs = n_outputs;
  m_layers.push_back(Layer("hidden",n_inputs,n_hiddens,false));
  m_layers.push_back(Layer("output",n_hiddens,n_outputs,true));
};

// destructor
NeuralNetwork::~NeuralNetwork() {
};

void NeuralNetwork::forward_propagate_inputs(vector<double>& inputs) {
  vector<double> new_inputs = inputs;
  for (int i=0;i<m_layers.size();i++) {
    Layer* layer_ptr = &m_layers[i];
    layer_ptr->evaluate(new_inputs);
    new_inputs = layer_ptr->get_outputs();
  }
}

void NeuralNetwork::back_propagate_errors(vector<double>& expected_outputs) {
  vector<double> layer_errors; // try to use layer.get_errors();
  // reverse iterate over layers
  for (int i=m_layers.size()-1;i>=0;i--) {
    layer_errors.clear();
    Layer* layer = &m_layers[i];
    // for each layer, iterate over nodes
    for (int j=0;j<layer->get_nInputs();j++) {
      double node_error = 0;
      // compute error of node
      if (layer->get_isOutputLayer()) {
        // output layer error is just "output - expected"
        node_error = layer->get_node(j).get_output() - expected_outputs[i];
      } else {
        // hidden layer error is "weighted error of next level projected back"
        Layer* next_layer = &m_layers[i+1];
        for (int k=0;k<next_layer->get_nNodes();k++)
          node_error += next_layer->get_node(k).get_delta() * next_layer->get_node(k).get_weight(k);
      }
      // save down results
      layer->get_node(j).set_delta(node_error);
      layer_errors.push_back(node_error);

      // debugging
      cout << layer->get_layerName() << endl;
      display_vector(layer_errors,"layer_errors");
      print();
    }
  }
}

void NeuralNetwork::update_weights() {
}

void NeuralNetwork::print(void) {
  cout << "network [" << this << "]"
  << ", nInputs = [" << m_nInputs << "]"
  << ", nHiddens = [" << m_nHiddens << "]"
  << ", nOutputs = [" << m_nOutputs << "]"
  << endl;
  for (auto layer : m_layers)
    layer.print();
}
