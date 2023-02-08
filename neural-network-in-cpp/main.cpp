#include "utils.hpp"
#include "Node.hpp"
#include "Layer.hpp"
#include "NeuralNetwork.hpp"
#include <iostream>
#include <array>    // for storing weights
#include <random>   // for initializing weights
using namespace std;

void Node_test_instantiation(void) {
  cout << __func__ << endl << endl;

  // instantiate
  Node n1 = Node("n1",2);
  Node n2 = Node("n2",3);
  n1.print();
  n2.print();
}

void Node_test_forward_propagation(void) {
  cout << __func__ << endl << endl;

  // instantiate
  Node n1 = Node("n1",2);
  cout << "activation = " << n1.activation_function(0) << endl;
  cout << "derivative = " << n1.transfer_derivative(0.5) << endl;

  // forward propagation
  vector<double> input1 = {1,2};
  n1.evaluate_inputs(input1);
  n1.print();
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
  vector<double> inputs = {1,2};
  l1.evaluate_inputs(inputs);
  l1.print();

  // compute deltas
  vector<double> expected_outputs = {3,4,5};
  l1.update_output_layer_deltas(expected_outputs);
  l1.print();

  // update weights
  l1.update_weights(inputs,0.5);
  l1.print();
}

void NeuralNetwork_test_instantiation(void) {
  cout << __func__ << endl << endl;

  NeuralNetwork nn1 = NeuralNetwork(3,2,4);
  NeuralNetwork nn2 = NeuralNetwork(2,2,5);
  nn1.print();
  nn2.print();
}

void NeuralNetwork_test_forward_propagation(void) {
  cout << __func__ << endl << endl;

  // instantiation
  NeuralNetwork nn1 = NeuralNetwork(3,2,4);
  nn1.print();

  // forward propagation
  vector<double> inputs = {1,2,3};
  nn1.forward_propagate_inputs(inputs);
  nn1.print();

  // back propagate errors
  vector<double> expected_outputs = {3,4,5,6};
  nn1.back_propagate_errors(expected_outputs);
  nn1.print();

  // update weights
  nn1.update_weights(inputs,0.5);
  nn1.print();
}

int main() {

  //Node_test_instantiation(); // instantiation
  //Node_test_forward_propagation(); // forward propagation

  //Layer_test_instantiation(); // instantiation
  //Layer_test_forward_propagation(); // forward propagation

  //NeuralNetwork_test_instantiation(); // instantiation
  //NeuralNetwork_test_forward_propagation(); // forward propagation


  // test instantiation
  // read input data

  // train it

  return 0;
}
