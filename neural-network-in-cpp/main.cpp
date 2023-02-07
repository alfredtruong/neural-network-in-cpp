#include "NeuralNetwork.hpp"
#include <iostream>
#include <array>    // for storing weights
#include <random>   // for initializing weights
using namespace std;

void Node_testing(void) {
  // instantiation testing
  cout << "test instantiation" << endl;
  Node n1 = Node("n1",2);
  //Node n2 = Node(3);
  //n1.print();
  //n2.print();

  cout << n1.activation_function(0) << endl;

  // forward propagation testing
  cout << "test forward propagation" << endl;
  vector<double> input1 = {1,2};
  n1.evaluate(input1);

  cout << n1.transfer_derivative() << endl;

  n1.print();
}

void Layer_testing(void) {
  // instantiation testing
  cout << "test instantiation" << endl;
  Layer l1 = Layer("l1",2,3,false);
  //Layer l2 = Layer("l2",10,2,false);
  l1.print();
  //l2.print();

  // forward propagation testing
  cout << "test forward propagation" << endl;
  vector<double> input1 = {1,2};
  l1.evaluate(input1);

  cout << endl;
  l1.print();
}

void NeuralNetwork_testing(void) {
  // instantiation testing
  cout << "test instantiation" << endl;
  NeuralNetwork nn1 = NeuralNetwork(3,2,4);
  //NeuralNetwork nn2 = NeuralNetwork(2,2,5);
  nn1.print();
  //nn2.print();

  // forward propagation testing
  cout << "test forward propagation" << endl;
  vector<double> input1 = {1,2,3};
  nn1.forward_propagate_inputs(input1);

  cout << endl;
  nn1.print();

  /*
  // back propagation error testing
  vector<double> input2 = {1,2,3};
  nn1.forward_propagate_inputs(input1);

  cout << endl;
  nn1.print();
  */

}

int main() {

  // testing
  //Node_testing();
  //Layer_testing();
  //NeuralNetwork_testing();

  // read input data

  // train it

  return 0;
}
