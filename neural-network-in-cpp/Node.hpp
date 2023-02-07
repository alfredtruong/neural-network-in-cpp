#ifndef NODE_H
#define NODE_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>

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

#endif
