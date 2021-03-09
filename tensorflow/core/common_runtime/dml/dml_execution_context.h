/* Copyright (c) Microsoft Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <condition_variable>
#include <functional>
#include <thread>
#include <vector>

#include "dml_command_allocator_ring.h"
#include "dml_command_list.h"
#include "dml_command_queue.h"
#include "dml_common.h"
#include "dml_descriptor_pool.h"
#include "dml_status.h"

namespace tensorflow {
class DmlAllocator;
class DmlCommandQueue;

// A thread-safe wrapper over DmlExecutionContextImpl. Calls to this class are
// batched to minimize work within the lock, and the batched calls are
// periodically flushed by a background thread (or by explicitly calling Flush).
class DmlExecutionContext {
 public:
  DmlExecutionContext(ID3D12Device* d3d12_device, IDMLDevice* dml_device,
                      ID3D12CommandQueue* queue, DmlAllocator* allocator);

  ~DmlExecutionContext();

  // NOTE: the caller is responsible for keeping the resources alive until the
  // returned GPU event has completed.
  DmlGpuEvent CopyBufferRegion(ID3D12Resource* dst_buffer, uint64_t dst_offset,
                               D3D12_RESOURCE_STATES dst_state,
                               ID3D12Resource* src_buffer, uint64_t src_offset,
                               D3D12_RESOURCE_STATES src_state,
                               uint64_t byte_count);

  // NOTE: the caller is responsible for keeping the resources alive until the
  // returned GPU event has completed.
  DmlGpuEvent FillBufferWithPattern(ID3D12Resource* dst, uint64_t dst_offset,
                                    uint64_t dst_size_in_bytes,
                                    absl::Span<const uint8_t> value);

  // NOTE: the caller is responsible for keeping the initializer and
  // descriptor_heap alive until the returned GPU event has completed.
  DmlGpuEvent InitializeOperator(
      IDMLOperatorInitializer* initializer,
      Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
      ID3D12DescriptorHeap* descriptor_heap);

  // NOTE: the caller is responsible for keeping the op and descriptor_heap
  // alive until the returned GPU event has completed.
  DmlGpuEvent ExecuteOperator(
      IDMLCompiledOperator* op,
      Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
      ID3D12DescriptorHeap* descriptor_heap);

  DmlGpuEvent ResourceBarrier(
      absl::Span<const D3D12_RESOURCE_BARRIER> barriers);

  // A slightly more efficient version of ResourceBarrier when the barrier span
  // only includes a UAV barrier (elides an extra copy).
  DmlGpuEvent UavBarrier();

  StatusOr<DmlGpuEvent> Flush();

  Status GetCommandRecorderStatus() const;

  DmlGpuEvent GetCurrentCompletionEvent();

  D3D12_COMMAND_LIST_TYPE GetCommandListTypeForQueue() const;

 private:
  static constexpr uint32_t default_batch_flush_size = 100;
  static constexpr uint32_t default_batch_flush_time_us = 1000;
  static constexpr uint32_t default_num_execution_threads = 1;

  using Command = std::function<void(DmlCommandList&)>;
  using Batch = absl::InlinedVector<Command, default_batch_flush_size>;

  // State that may be accessed or modified by threads that call the
  // execution context as well as the background execution thread(s).
  struct SharedState {
    std::mutex mutex;
    DmlGpuEvent next_flush_event;
    std::condition_variable new_function_enqueued;

    // Commands are double buffered: callers extend the "write batch" while the
    // background thread flushes the "execute batch".
    Batch batches[2];
    uint32_t write_batch_index = 0;
    Batch& WriteBatch() { return batches[write_batch_index]; }

    bool exit_requested = false;
    bool flush_requested = false;

    Status status;
  };

  std::shared_ptr<SharedState> shared_state_;

  std::shared_ptr<DmlCommandQueue> dml_command_queue_;
  absl::InlinedVector<DmlCommandList, default_num_execution_threads>
      command_lists_;

  // TODO: exec_threads_[0] is the main execution thread that swaps the buffers.
  // Others just record.
  std::thread thread_;
  // absl::InlinedVector<std::thread, default_num_execution_threads>
  //     exec_threads_;

  static void ThreadProc(std::shared_ptr<SharedState> state,
                         DmlCommandQueue* dml_command_queue,
                         absl::Span<DmlCommandList> dml_command_lists,
                         uint32_t batch_flush_size,
                         uint32_t batch_flush_time_us);

  static void RecordCommands(absl::Span<Command> commands,
                             DmlCommandList& command_list);
};

}  // namespace tensorflow
