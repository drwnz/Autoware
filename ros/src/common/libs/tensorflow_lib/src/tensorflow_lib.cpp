/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ********************
 *  v1.0: drwnz (david.wong@tier4.jp)
 *
 * tensorflow_lib.cpp
 *
 *  Created on: February 6th 2018
 */

// #include <cstdio>
// #include <stdio.h>
// #include <cstdlib>
#include <tensorflow_lib/tensorflow_lib.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace tensorflow
{

TensorFlowSession::TensorFlowSession(const char* graph_filename, const char* input_operation_name, const char* output_operation_name)
{
  graph_ = TF_NewGraph();
  status_ = TF_NewStatus();
  input_tensors_.clear();
  output_tensors_ = {nullptr};

  TF_ImportGraphDefOptions* graph_options = TF_NewImportGraphDefOptions();
  TF_SessionOptions* session_options = TF_NewSessionOptions();

  TF_Buffer* buffer = ReadBufferFromFile(graph_filename);
  if (buffer == nullptr) {
    std::cout << "A" << std::endl; // Error handling.
  }

  TF_GraphImportGraphDef(graph_, buffer, graph_options, status_);
  TF_DeleteImportGraphDefOptions(graph_options);
  TF_DeleteBuffer(buffer);

  if (TF_GetCode(status_) != TF_OK) {
    TF_DeleteGraph(graph_);
    //Error handling.
    graph_ = nullptr;
    std::cout << "B" << std::endl;
  }

  session_ = TF_NewSession(graph_, session_options, status_);
  TF_DeleteSessionOptions(session_options);
  if (TF_GetCode(status_) != TF_OK) {
    // Error handling.
    std::cout << "C" << std::endl;
  }
  input_operation_ = {{TF_GraphOperationByName(graph_, input_operation_name), 0}};
  output_operation_ = {{TF_GraphOperationByName(graph_, output_operation_name), 0}};
}

TensorFlowSession::~TensorFlowSession()
{
  std::cout << "0" << std::endl;
  if (output_tensors_.size() > 0) {
    std::cout << output_tensors_.size() << std::endl;
    DeleteTensorVector(output_tensors_);
  }
  std::cout << "A" << std::endl;
  if (input_tensors_.size() > 0) {
    DeleteTensorVector(input_tensors_);
  }
  std::cout << "B" << std::endl;
  TF_CloseSession(session_, status_);
  std::cout << "C" << std::endl;
  TF_DeleteSession(session_, status_);
  std::cout << "D" << std::endl;
  TF_DeleteStatus(status_);
  std::cout << "E" << std::endl;
  TF_DeleteGraph(graph_);
  std::cout << "F" << std::endl;
}

bool TensorFlowSession::AddInputTensor(TF_DataType input_tf_type, const std::vector<std::int64_t>& input_dimensions, const void* input_data, std::size_t data_length)
{
  if (input_dimensions.data() == nullptr || input_data == nullptr) {
    std::cout << "D" << std::endl;
    return false;
  }

  TF_Tensor* input_tensor = TF_AllocateTensor(input_tf_type, input_dimensions.data(), static_cast<int>(input_dimensions.size()), data_length);

  if (input_tensor == nullptr) {
    std::cout << "E" << std::endl;
    return false;
  }

  void* input_tensor_data = TF_TensorData(input_tensor);
  if (input_tensor_data == nullptr) {
    TF_DeleteTensor(input_tensor);
    std::cout << "F" << std::endl;
    return false;
  }
  // Try a direct pointer swap here later.
  std::memcpy(input_tensor_data, input_data, std::min(data_length, TF_TensorByteSize(input_tensor)));
  input_tensors_.push_back(input_tensor);
  return true;
}

bool TensorFlowSession::RunInference(void)
{
  if (input_operation_.data() == nullptr || output_operation_.data() == nullptr) {
    std::cout << "G" << std::endl;
    return false;
  }

  if (input_tensors_.size() == 0) {
    std::cout << "H" << std::endl;
    return false;
  }
  // if (output_tensors_.data() != 0) {
  //   std::cout << "H.1" << std::endl;
  //   return false;
  // }
  // Important - delete old tensors!
  if (output_tensors_.size() > 0) {
    DeleteTensorVector(output_tensors_);
  }
  output_tensors_ = {nullptr};
  std::cout << "Got 4.1" << std::endl;
  std::cout << output_tensors_.size() << std::endl;
  TF_SessionRun(session_,
              nullptr, // Run options.
              input_operation_.data(), input_tensors_.data(), static_cast<int>(input_tensors_.size()), // Input tensors, input tensor values, number of inputs.
              output_operation_.data(), output_tensors_.data(), static_cast<int>(output_tensors_.size()), // Output tensors, output tensor values, number of outputs.
              nullptr, 0, // Target operations, number of targets.
              nullptr, // Run metadata.
              status_ // Output status.
  );

  std::cout << "Got 4.2" << std::endl;
  std::cout << input_tensors_.size() << std::endl;
  // Careful here - delete reference to data only?
  DeleteTensorVector(input_tensors_);
  std::cout << input_tensors_.size() << std::endl;
  std::cout << "Got 4.3" << std::endl;
  if (TF_GetCode(status_) != TF_OK) {
    DeleteTensorVector(output_tensors_);
    DeleteTensorVector(input_tensors_);
    std::cout << "I" << std::endl;
    return false;
  }
  return true;
}

void TensorFlowSession::DeleteTensorVector(std::vector<TF_Tensor*>& tensor_vector)
{
  for (auto tensor : tensor_vector) {
    if (tensor != nullptr) {
      TF_DeleteTensor(tensor);
    }
  }
  tensor_vector.clear();
}

void DeallocateBuffer(void* data, size_t)
{
  std::free(data);
}

TF_Buffer* TensorFlowSession::ReadBufferFromFile(const char* filename)
{
  const auto file = std::fopen(filename, "rb");
  if (file == nullptr) {
    std::cout << "J" << std::endl;
    return nullptr;
  }

  std::fseek(file, 0, SEEK_END);
  const auto filesize = ftell(file);
  std::fseek(file, 0, SEEK_SET);

  if (filesize < 1) {
    std::fclose(file);
    std::cout << "K" << std::endl;
    return nullptr;
  }

  const auto data = std::malloc(filesize);
  std::fread(data, filesize, 1, file);
  std::fclose(file);

  TF_Buffer* buffer = TF_NewBuffer();
  buffer->data = data;
  buffer->length = filesize;
  buffer->data_deallocator = DeallocateBuffer;

  return buffer;
}
} // namespace tensorflow
