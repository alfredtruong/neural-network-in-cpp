#ifndef LAYER_H
#define LAYER_H

#include "Node.hpp"
#include <vector>
#include <string>

using namespace std;

class Layer
{
private:
  string m_layerName;          // layer label
  int m_nInputs;               // counter of inputs
  int m_nNodes;                // counter of nodes
  vector<Node> m_nodes;        // look into "operator[] overloading for vector"
  vector<double> m_output;     // intermediate container for layer output

public:
  Layer(string layer_name,int n_inputs,int n_nodes); // constructor
  ~Layer();                                          // destructor

  void evaluate_input(const vector<double>& input);

  // update node deltas
  void update_output_layer_deltas(const vector<double>& expected_output);
  void update_hidden_layer_deltas(Layer& next_layer);

  // update node weights
  void update_weights(const vector<double>& input,double learning_rate); // update weights of all nodes in layer

  // getters
  string get_layerName(void) { return m_layerName; };
  int get_nInputs(void) { return m_nInputs; };
  int get_nNodes(void) { return m_nNodes; };
  Node get_node(int node_idx) { return m_nodes[node_idx]; }
  vector<double> get_output(void) { return m_output; };

  // print
  void print(void);
};

void Layer_test_instantiation(void);
void Layer_test_forward_propagation(void);

#endif
