#include "Node.hpp"
#include <iostream>
#include <cmath>
#include <random>

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

// gradient of activation function
double Node::transfer_derivative(double x) {
  return x * (1.0 - x);
};

void Node::compute_activation(const vector<double>& inputs) {
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
void Node::evaluate_inputs(const vector<double>& inputs) {
  compute_activation(inputs);
  compute_output();
  //cout << "[" << this << "][" << __func__ << "] " << endl;
  //this->print();
}

void Node::print(void) {
  cout << "node [" << this << "]";
  cout << ", name = [" << m_nodeName << "]";
  cout << ", nWeights = [" << m_nWeights << "]";
  cout << endl;

  cout << "\t";
  cout << "bias = [ " << m_weights[m_nWeights] << " ], ";
  cout << "weights = [ ";
  for (int i=0;i<m_nWeights;i++) cout << m_weights[i] << ", ";
  cout << "]" << endl;

  cout << "\t";
  cout << "activation = [" << m_activation << "], ";
  cout << "output = [" << m_output << "], ";
  cout << "delta = [" << m_delta << "]" << endl;

  cout << endl;
}

/*
ostream& operator<< (ostream& os, const Node& node) {
  //os << node.get_weights();
  os << node.get_delta();
  os << node.get_output;
  return os;
}
*/
