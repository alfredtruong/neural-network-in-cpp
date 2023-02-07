#ifndef LAYER_H
#define LAYER_H

#include "Node.hpp"

using namespace std;

class Layer
{
private:
  string m_layerName;
  bool m_isOutputLayer;
  int m_nInputs;
  int m_nNodes;
  vector<Node> m_nodes;
  vector<double> m_outputs;
  vector<double> m_errors;

public:
  Layer(string layer_name,int n_inputs,int n_nodes,bool is_output_layer); // constructor
  ~Layer();                        // destructor


  void evaluate(vector<double>& inputs);
  /*
  void back_propagation_errors(vector<double>& errors); // compute errors of all nodes in layer
  void update_weights();                                // update weights of all nodes in layer
  */

  // getters
  string get_layerName(void) { return m_layerName; };
  bool get_isOutputLayer(void) { return m_isOutputLayer; };
  int get_nInputs(void) { return m_nInputs; };
  int get_nNodes(void) { return m_nNodes; };
  Node get_node(int node_idx) { return m_nodes[node_idx]; }
  vector<double> get_outputs(void) { return m_outputs; };
  vector<double> get_errors(void) { return m_errors; };

  // print
  void print(void);
};

#endif
