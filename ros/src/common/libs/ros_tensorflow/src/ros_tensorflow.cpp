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
 * ros_tensorflow.cpp
 *
 *  Created on: February 6th 2019
 */

#include <ros_tensorflow/ros_tensorflow.hpp>
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
    // Error handling.
  }

  TF_GraphImportGraphDef(graph_, buffer, graph_options, status_);
  TF_DeleteImportGraphDefOptions(graph_options);
  TF_DeleteBuffer(buffer);

  if (TF_GetCode(status_) != TF_OK) {
    TF_DeleteGraph(graph_);
    // Error handling.
    graph_ = nullptr;
  }

  session_ = TF_NewSession(graph_, session_options, status_);
  TF_DeleteSessionOptions(session_options);
  if (TF_GetCode(status_) != TF_OK) {
    // Error handling.
  }
  input_operation_ = {{TF_GraphOperationByName(graph_, input_operation_name), 0}};
  output_operation_ = {{TF_GraphOperationByName(graph_, output_operation_name), 0}};
}

TensorFlowSession::~TensorFlowSession()
{
  if (output_tensors_.size() > 0) {
    std::cout << output_tensors_.size() << std::endl;
    DeleteTensorVector(output_tensors_);
  }
  if (input_tensors_.size() > 0) {
    DeleteTensorVector(input_tensors_);
  }
  TF_CloseSession(session_, status_);
  TF_DeleteSession(session_, status_);
  TF_DeleteStatus(status_);
  TF_DeleteGraph(graph_);
}

bool TensorFlowSession::AddInputTensor(TF_DataType input_tf_type, const std::vector<std::int64_t>& input_dimensions, const void* input_data, std::size_t data_length)
{
  if (input_dimensions.data() == nullptr || input_data == nullptr) {
    return false;
  }

  TF_Tensor* input_tensor = TF_AllocateTensor(input_tf_type, input_dimensions.data(), static_cast<int>(input_dimensions.size()), data_length);

  if (input_tensor == nullptr) {
    return false;
  }

  void* input_tensor_data = TF_TensorData(input_tensor);
  if (input_tensor_data == nullptr) {
    TF_DeleteTensor(input_tensor);
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
    return false;
  }

  if (input_tensors_.size() == 0) {
    return false;
  }

  if (output_tensors_.size() > 0) {
    DeleteTensorVector(output_tensors_);
  }
  output_tensors_ = {nullptr};
  std::cout << output_tensors_.size() << std::endl;
  TF_SessionRun(session_,
              nullptr, // Run options.
              input_operation_.data(), input_tensors_.data(), static_cast<int>(input_tensors_.size()), // Input tensors, input tensor values, number of inputs.
              output_operation_.data(), output_tensors_.data(), static_cast<int>(output_tensors_.size()), // Output tensors, output tensor values, number of outputs.
              nullptr, 0, // Target operations, number of targets.
              nullptr, // Run metadata.
              status_ // Output status.
  );

  std::cout << input_tensors_.size() << std::endl;
  // Careful here - delete reference to data only?
  DeleteTensorVector(input_tensors_);
  std::cout << input_tensors_.size() << std::endl;
  if (TF_GetCode(status_) != TF_OK) {
    DeleteTensorVector(output_tensors_);
    DeleteTensorVector(input_tensors_);
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
    return nullptr;
  }

  std::fseek(file, 0, SEEK_END);
  const auto filesize = ftell(file);
  std::fseek(file, 0, SEEK_SET);

  if (filesize < 1) {
    // Error handling
    std::fclose(file);
    return nullptr;
  }

  const auto data = std::malloc(filesize);
  const auto readsize = std::fread(data, filesize, 1, file);
  if (readsize < 1) {
    // Error handling
    std::free(data);
    std::fclose(file);
    return nullptr;
  }

  std::fclose(file);

  TF_Buffer* buffer = TF_NewBuffer();
  buffer->data = data;
  buffer->length = filesize;
  buffer->data_deallocator = DeallocateBuffer;

  return buffer;
}
} // namespace tensorflow
