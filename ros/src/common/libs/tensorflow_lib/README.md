# TensorFlow interface library

This Autoware package provides a simple interface to the TensorFlow C API.

### Requirements

* NVIDIA GPU with CUDA installed. The pre-compiled TensorFlow C API requires a CPU with compute capability of 6.0 or higher.
* [TensorFlow C API](https://www.tensorflow.org/install/lang_c) installed. The [Linux GPU version](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.12.0.tar.gz) is recommended, as long as you have CUDA installed.

### How to install the TensorFlow C API

Detailed instructions coming soon...

### Usage

This library implements a class,`TensorFlowSession` which loads a TensorFlow graph. Here is a minimal example to illustrate usage:

```cpp
#include <tensorflow_lib/tensorflow_lib.hpp>

/*
Create a new session:
"test_graph.pb"        const char*    Frozen graph file containing model and weights
"input_operation"      const char*    Name of input operation within the graph
"output operation"     const char*    Name of output operation within the graph

This will load the graph and start the TensorFlow session.
*/
TensorFlowSession test_session = TensorFlowSession("test_graph.pb", "input_operation", "output_operation");

/*
Create input tensor which match the inputs of the loaded graph:
input_data_vector      std::vector<input_type>&    Flattened vector of input values
input_data_dimensions  std::vector<std::int64_t>&  Vector of dimensions of the input

Multiple input tensors can be added by calling this function multiple times, noting that order is important.
All created input tensors will be deallocated after inference.
*/
test_session.AddInputVector(input_data_vector, input_data_dimensions);

/*
Run inference using on the input tensors:
success                 bool    True if successful, False if failed

This function also clears the input tensor,ready for the next input.
*/
success = test_session.RunInference();

/*
Retrieve the output data vectors:
output_data_vector      std::vector<std::vector<output_type>>&    Vector of flattened output vectors

Output vectors are flattened.
*/
test_session.GetOutputVectors(output_data_vectors);
```
