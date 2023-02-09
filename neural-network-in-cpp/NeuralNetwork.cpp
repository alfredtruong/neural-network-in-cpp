#include "NeuralNetwork.hpp"
#include "utils.hpp"
#include <iostream>

////////////////////////////////////////////////////////
// NETWORK
////////////////////////////////////////////////////////

// initialize network
NeuralNetwork::NeuralNetwork(int n_inputs,int n_hiddens,int n_outputs) {
  m_nInputs = n_inputs;
  m_nHiddens = n_hiddens;
  m_nOutputs = n_outputs;

  m_nLayers = 0;
  add_layer("hidden",n_inputs,n_hiddens);
  add_layer("output",n_hiddens,n_outputs);
};

// destructor
NeuralNetwork::~NeuralNetwork() {
};

// add layer
void NeuralNetwork::add_layer(string name,int n_inputs,int n_nodes) {
  m_layers.push_back(Layer(name,n_inputs,n_nodes));
  m_nLayers++;
};

// forward iterate over layers
void NeuralNetwork::forward_propagate_input(const vector<double>& input) {
  vector<double> new_input = input;
  for (int i=0;i<m_nLayers;i++) {
    Layer& this_layer = m_layers[i];
    this_layer.evaluate_input(new_input);
    new_input = this_layer.get_output(); // next layer looks at this layers output
  }
  return new_input;
}

// reverse iterate over layers
void NeuralNetwork::back_propagate_errors(const vector<double>& expected_output) {
  for (int i=m_nLayers-1;i>=0;i--) {
    Layer& this_layer = m_layers[i];
    if (int i=m_nLayers-1) {
      this_layer.update_output_layer_deltas(expected_output); // output layer error = "output - expected"
    } else {
      Layer& next_layer = m_layers[i+1];
      this_layer.update_hidden_layer_deltas(next_layer);       // hidden layer error = "weighted error of next level projected back"
    }
  }
}

// forward iterate over layers
void NeuralNetwork::update_weights(const vector<double>& input, double learning_rate) {
  vector<double> new_input = input;
  for (int i=0;i<m_nLayers;i++) {
    Layer& this_layer = m_layers[i];
    this_layer.update_weights(new_input,learning_rate);
    new_input = this_layer.get_output(); // next layer looks at this layers output
  }
}

void NeuralNetwork::print(void) {
  cout << "network [" << this << "]";
  cout << ", nInputs = [" << m_nInputs << "]";
  cout << ", nHiddens = [" << m_nHiddens << "]";
  cout << ", nOutputs = [" << m_nOutputs << "]";
  cout << endl;
  for (auto layer : m_layers) layer.print();
  cout << endl;
}

void NeuralNetwork::train_single_observation(classification_observation& observation, double learning_rate) {
  // prep input attributes
  vector<double>& input = observation.second;

  // prep input class
  observation_class actual_classID = observation.first;
  vector<double>& expected_output = classID_to_onehot(actual_classID);

  // fit for single observation
  forward_propagate_input(input);
  back_propagate_errors(expected_output);
  update_weights(input,learning_rate);
}

void NeuralNetwork::train_entire_dataset(classification_dataset& dataset,double learning_rate) {
  // send entire dataset through nnet
  for (auto& observation : dataset)
    train_single_observation(classification_observation observation,double learning_rate)
}

void NeuralNetwork::train_n_epochs(classification_dataset& dataset,double learning_rate, int n_epochs,bool verbose,int verbose_epochs) {
  // train entire dataset n_epochs times
  for (int i=0;i<n_epochs;i++) {
    train_entire_dataset(dataset,learning_rate);

    // evaluate errors
    if (verbose)
      if (i%verbose_epochs==0) {

      }

  }
}

void NeuralNetwork::evaluate_network(classification_dataset& dataset) {
  double sum_squared_errors = 0;
  double accuracy = 0;
  int correct_counter = 0;
  for (auto& observation : dataset) {
    // get prediction vector
    vector<double> output = predict(observation.second);
    // get class prediction
    observation_class predicted_classID = output_to_classID(output);

    // count correct predictions
    if (observation.first) == predicted_classID)
      correct_counter++;

    vector<double> expected_output = classID_to_onehot(observation.first);
    // compute squared errors
    sum_squared_errors += output_squared_error(output,)
  }
  return static_cast<double>(correct) / predicted.size();




  }

vector<double> NeuralNetwork::predict(const vector<double>& input) {
  forward_propagate_input(input);
  Layer& output_layer = m_layers[m_nLayers-1];
  return output_layer.get_output();
}

observation_class NeuralNetwork::output_to_classID(const vector<double>& output);
  return argmax(output);
}

// nClasses is N, the count of unique classes in the dataset
// classID integer in [0,nClasses) denoting which class an observation is from
vector<double> NeuralNetwork::classID_to_onehot(observation_class classID) {
  vector<double> onehot(m_nOutputs,0);
  onehot[classID] = 1.0;
  return onehot;
}

// distance(single observation class probability prediction,expected_output)
double NeuralNetwork::output_squared_error(vector<double> output,vector<double> expected_output) {
  double squared_error = 0;
  for (int i=0;i<m_nOutputs;i++) {
    double error = (output[i] - expected_output[i]);
    squared_error += error * error;
  }
  return squared_error;
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
  vector<double> input = {1,2,3};
  nn1.forward_propagate_input(input);
  nn1.print();

  // back propagate errors
  vector<double> expected_output = {3,4,5,6};
  nn1.back_propagate_errors(expected_output);
  nn1.print();

  // update weights
  nn1.update_weights(input,0.5);
  nn1.print();
}
