#include "NeuralNetwork.hpp"
#include <cmath>
#include <random>
#include <iostream>

////////////////////////////////////////////////////////
// UTILS
////////////////////////////////////////////////////////
template <typename T>
void display_vector(vector<T>& v,const string name) {
  // show outputs
  cout << "\t[" << name << "] n = [" << v.size() << "], data = [ ";
  for (auto x: v) cout << x << ", ";
  cout << "]" << endl;
}

////////////////////////////////////////////////////////
// NODE
////////////////////////////////////////////////////////

// initialize node
Node::Node(string node_name,int n_inputs) {
  m_nodeName = node_name;
  m_nWeights = n_inputs;

  // initialize node weights + bias (<=)
  for (int i=0;i<=m_nWeights;i++)
    m_weights.push_back(static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

  // initialize forward-propagation fields
  m_activation = 0;

  // initialize back-propagation fields
  m_output = 0;
  m_delta = 0;
};

// destructor
Node::~Node() {
};

// computes transfer for given activation
double Node::activation_function(double x) {
  return 1.0 / (1.0 + exp(-x));
};

// gives gradient at m_output
double Node::transfer_derivative(void) {
  return m_output * (1.0 - m_output);
};

void Node::compute_activation(vector<double>& inputs) {
  m_activation = m_weights[m_nWeights]; // overwriting existing value, start with bias
  for (int i=0;i<m_nWeights;i++)
    m_activation += m_weights[i] * inputs[i]; // add weight x input
  //cout << "[" << this << "][" << __func__ << "] " << m_activation << endl;
}

void Node::compute_output(void) {
  m_output = activation_function(m_activation);
  //cout << "[" << this << "][" << __func__ << "] " << m_output << endl;
}

// forward propagate inputs
void Node::evaluate(vector<double>& inputs) {
  compute_activation(inputs);
  compute_output();
  //cout << "[" << this << "][" << __func__ << "] " << endl;
  //this->print();
}

void Node::print(void) {
  cout << "node [" << this << "]"
  << ", name = [" << m_nodeName << "]"
  << ", nWeights = [" << m_nWeights << "]"
  << endl;

  cout << "\tbias = [ " << m_weights[m_nWeights] << " ], ";
  cout << "weights = [ ";
  for (int i=0;i<m_nWeights;i++) cout << m_weights[i] << ", ";
  cout << "]" << endl;

  cout << "\tactivation = [" << m_activation << "], ";
  cout << "output = [" << m_output << "], ";
  cout << "delta = [" << m_delta << "]" << endl;
}
/*
ostream& operator<< (ostream& os, const Node& node) {
  //os << node.get_weights();
  os << node.get_delta();
  os << node.get_output;
  return os;
}
*/

////////////////////////////////////////////////////////
// LAYER
////////////////////////////////////////////////////////

// initialize layer
Layer::Layer(string layer_name,int n_inputs,int n_nodes,bool is_output_layer) {
  m_layerName = layer_name;
  m_isOutputLayer = is_output_layer;
  m_nInputs = n_inputs;
  m_nNodes = n_nodes;
  for (int i=0;i<m_nNodes;i++)
    m_nodes.push_back(Node(m_layerName,m_nInputs));
};

// destructor
Layer::~Layer() {
};

// forward propagate inputs
void Layer::evaluate(vector<double>& inputs) {
  m_outputs.clear();
  for (size_t i=0;i<m_nNodes;i++) {
    Node* node_ptr = &m_nodes[i];
    node_ptr->evaluate(inputs);
    m_outputs.push_back(node_ptr->get_output());
  }
}

void Layer::print(void) {
  cout << "layer [" << this << "]"
  << ", name = [" << m_layerName << "]"
  << ", nInputs = [" << m_nInputs << "]"
  << ", nNodes = [" << m_nNodes << "]" << endl;

  display_vector(m_outputs,"outputs");             // display outputs
  display_vector(m_errors,"errors");               // display errors
  for (int i=0;i<m_nNodes;i++) m_nodes[i].print(); // display nodes
}

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
