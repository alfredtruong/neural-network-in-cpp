#include "Layer.hpp"
#include "utils.hpp"

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

