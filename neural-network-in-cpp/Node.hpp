#ifndef NODE_H
#define NODE_H

#include <vector>
#include <string>

using namespace std;

class Node
{
private:
  string m_nodeName;          // node label
  int m_nWeights;             // counter of weights
  vector<double> m_weights;   // weights + bias
  double m_activation;        // intermediate container for node activation given inputs
  double m_output;            // intermediate container for node outputs
  double m_delta;             // intermediate container for node weight update deltas

public:
  Node(string node_name,int n_inputs);    // constructor
  ~Node();                                // destructor

  double activation_function(double x);   // computes transfer for given activation
  double transfer_derivative(double x);   // gives gradient at m_output

  void compute_activation(const vector<double>& inputs); // compute activation value given inputs
  void compute_output(void);                             // activation function applied to activation value
  void evaluate_inputs(const vector<double>& inputs);


  // getters
  int get_nWeights(void) { return m_nWeights; };
  double get_weight(int weight_idx) { return m_weights[weight_idx]; };
  double get_output(void) { return m_output; };
  double get_delta(void) { return m_delta; };

  // setters
  void set_delta(double delta) { m_delta = delta; }
  void set_weight(int weight_idx,double new_weight) { m_weights[weight_idx] = new_weight; }

  // print
  void print(void);

  //friend std::ostream& operator<< (std::ostream& out, const Node& node);
};

#endif
