#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#import <vector>
#import <ostream>

using namespace std;

class Node
{
private:
  string m_nodeName;
  int m_nWeights;
  vector<double> m_weights;   // weights + bias
  double m_activation;        // activation given inputs
  double m_output;            // transfer of activation
  double m_delta;             // own error given error in subsequent layer

public:
  Node(string node_name,int n_inputs);    // constructor
  ~Node();               // destructor

  double activation_function(double x);  // computes transfer for given activation
  double transfer_derivative(void);      // gives gradient at m_output

  void compute_activation(vector<double>& inputs); // compute activation value given inputs
  void compute_output(void);                       // activation function applied to activation value
  void evaluate(vector<double>& inputs);


  // getters
  double get_weight(int weight_idx) { return m_weights[weight_idx]; }
  double get_output() { return m_output; };
  double get_delta() { return m_delta; };

  // setters
  void set_delta(double delta) { m_delta = delta; }
  void set_weight(int weight_idx,double new_weight) { m_weights[weight_idx] = new_weight; }

  // print
  void print(void);

  //friend std::ostream& operator<< (std::ostream& out, const Node& node);
};

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
