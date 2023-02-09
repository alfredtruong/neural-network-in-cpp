#include "Layer.hpp"
#include "utils.hpp"

////////////////////////////////////////////////////////
// LAYER
////////////////////////////////////////////////////////

// initialize layer
Layer::Layer(string layer_name,int n_inputs,int n_nodes) {
  m_layerName = layer_name;
  m_nInputs = n_inputs;
  m_nNodes = n_nodes;
  for (int i=0;i<m_nNodes;i++)
    m_nodes.push_back(Node(m_layerName,m_nInputs));
};

// destructor
Layer::~Layer() {
};

// forward propagate input
void Layer::evaluate_input(const vector<double>& input) {
  m_output.clear();
  for (size_t i=0;i<m_nNodes;i++) {
    Node& this_node = m_nodes[i];
    this_node.evaluate_input(input);
    m_output.push_back(this_node.get_output());
  }
}

// compute delta on output layer
void Layer::update_output_layer_deltas(const vector<double>& expected_output) {
  for (int i=0;i<m_nNodes;i++) {
    Node& this_node = m_nodes[i];
    // compute error = "output - expected"
    double tmp = this_node.get_output() - expected_output[i];
    // compute delta
    tmp = tmp * this_node.transfer_derivative(this_node.get_output());
    this_node.set_delta(tmp);
  }
}

// compute delta on hidden layer
void Layer::update_hidden_layer_deltas(Layer& next_layer) {
  for (int i=0;i<m_nNodes;i++) {
    Node& this_node = m_nodes[i];
    // compute error = "weighted error of next level projected back"
    double tmp = 0;
    for (int j=0;j<next_layer.get_nInputs();j++) {
      Node next_node = next_layer.get_node(j);
      tmp += next_node.get_delta() * next_node.get_weight(j);
    }
    // compute delta
    tmp *= this_node.transfer_derivative(this_node.get_output());
    this_node.set_delta(tmp);
  }
}

// update weights for each node
void Layer::update_weights(const vector<double>& input, double learning_rate) {
  for (int i=0;i<m_nNodes;i++) {
    Node& this_node = m_nodes[i];
    double tmp; // working container
    // node bias
    tmp = this_node.get_weight(this_node.get_nWeights());
    tmp -= learning_rate * this_node.get_delta();
    this_node.set_weight(this_node.get_nWeights(),tmp); // investigate operator[] overloading
    // node weights
    for (int j=0;j<this_node.get_nWeights();j++) {
      tmp = this_node.get_weight(j);
      tmp -= learning_rate * this_node.get_delta() * input[j];
      this_node.set_weight(j,tmp); // investigate operator[] overloading
    }
  }

}

void Layer::print(void) {
  cout << "layer [" << this << "]"
  << ", name = [" << m_layerName << "]"
  << ", nInputs = [" << m_nInputs << "]"
  << ", nNodes = [" << m_nNodes << "]" << endl;

  display_vector(m_output,"output");               // display output
  //display_vector(m_errors,"error");              // display error
  for (int i=0;i<m_nNodes;i++) m_nodes[i].print(); // display nodes

  cout << endl;
}

void Layer_test_instantiation(void) {
  cout << __func__ << endl << endl;

  // instantiate
  Layer l1 = Layer("l1",2,3);
  Layer l2 = Layer("l2",10,2);
  l1.print();
  l2.print();
}

void Layer_test_forward_propagation(void) {
  cout << __func__ << endl << endl;

  // instantiate
  Layer l1 = Layer("l1",2,3);
  l1.print();

  // forward propagation
  vector<double> input = {1,2};
  l1.evaluate_input(input);
  l1.print();

  // compute deltas
  vector<double> expected_output = {3,4,5};
  l1.update_output_layer_deltas(expected_output);
  l1.print();

  // update weights
  l1.update_weights(input,0.5);
  l1.print();
}
