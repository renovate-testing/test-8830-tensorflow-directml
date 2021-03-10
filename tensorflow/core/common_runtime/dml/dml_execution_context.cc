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

#include "dml_execution_context.h"

#include "dml_bfc_allocator.h"
#include "dml_buffer.h"
#include "dml_tracing.h"
#include "dml_util.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

DmlExecutionContext::DmlExecutionContext(ID3D12Device* d3d_device,
                                         IDMLDevice* dml_device,
                                         ID3D12CommandQueue* queue,
                                         DmlAllocator* allocator) {
  dml_command_queue_ = std::make_shared<DmlCommandQueue>(queue);

  batch_state_ = std::make_shared<BatchState>();
  batch_state_->next_flush_event =
      dml_command_queue_->GetCurrentCompletionEvent();
  ++batch_state_->next_flush_event.fence_value;

  uint32_t batch_flush_size = default_batch_flush_size;
  {
    int64 batch_flush_size_int64 = 0;
    Status s = ReadInt64FromEnvVar("TF_DIRECTML_BATCH_FLUSH_SIZE", 0,
                                   &batch_flush_size_int64);
    if (s.ok() && batch_flush_size_int64 != 0) {
      batch_flush_size = static_cast<uint32_t>(batch_flush_size_int64);
    }
  }

  uint32_t batch_flush_time_us = default_batch_flush_time_us;
  {
    int64 batch_flush_time_us_int64 = 0;
    Status s = ReadInt64FromEnvVar("TF_DIRECTML_BATCH_FLUSH_TIME", 0,
                                   &batch_flush_time_us_int64);
    if (s.ok() && batch_flush_time_us_int64 != 0) {
      batch_flush_time_us = static_cast<uint32_t>(batch_flush_time_us_int64);
    }
  }

  uint32_t num_exec_threads = default_num_exec_threads;
  {
    int64 num_exec_threads_int64 = 0;
    Status s = ReadInt64FromEnvVar("TF_DIRECTML_EXECUTION_THREADS", 0,
                                   &num_exec_threads_int64);
    if (s.ok() && num_exec_threads_int64 != 0) {
      num_exec_threads = static_cast<uint32_t>(num_exec_threads_int64);
    }
  }

  exec_state_ = std::make_shared<ExecutionState>();
  exec_state_->command_starts.resize(num_exec_threads);
  exec_state_->command_counts.resize(num_exec_threads);

  for (uint32_t thread_id = 0; thread_id < num_exec_threads; thread_id++) {
    exec_state_->command_lists.emplace_back(
        d3d_device, dml_device, dml_command_queue_->GetType(), allocator);

    // The first thread is the main execution thread, which drives the other
    // threads (if any) and submits to the hardware.
    if (thread_id == 0) {
      exec_threads_.emplace_back(MainExecutionThreadProc, batch_state_,
                                 exec_state_, dml_command_queue_.get(),
                                 batch_flush_size, batch_flush_time_us);
    } else {
      exec_threads_.emplace_back(SecondaryExecutionThreadProc, thread_id,
                                 exec_state_);
    }
  }
}

DmlExecutionContext::~DmlExecutionContext() {
  // Request exit of the background thread
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  batch_state_->exit_requested = true;
  batch_state_->new_function_enqueued.notify_all();  // wake the thread
  lock.unlock();

  // detach() rather than join(), because we don't want (or need) to wait for
  // it to complete. This prevents blocking in a destructor, which would be
  // bad.
  for (auto& thread : exec_threads_) {
    thread.detach();
  }
}

DmlGpuEvent DmlExecutionContext::CopyBufferRegion(
    ID3D12Resource* dst_buffer, uint64_t dst_offset,
    D3D12_RESOURCE_STATES dst_state, ID3D12Resource* src_buffer,
    uint64_t src_offset, D3D12_RESOURCE_STATES src_state, uint64_t byte_count) {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  VLOG(3) << "DML EC: Batch CopyBufferRegion with fv = "
          << batch_state_->next_flush_event.fence_value;

  batch_state_->WriteBatch().emplace_back([=](DmlCommandList& command_list) {
    command_list.CopyBufferRegion(dst_buffer, dst_offset, dst_state, src_buffer,
                                  src_offset, src_state, byte_count);
  });

  batch_state_->new_function_enqueued.notify_all();

  return batch_state_->next_flush_event;
}

DmlGpuEvent DmlExecutionContext::FillBufferWithPattern(
    ID3D12Resource* dst, uint64_t dst_offset, uint64_t dst_size_in_bytes,
    absl::Span<const uint8_t> value) {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  VLOG(3) << "DML EC: Batch FillBufferWithPattern with fv = "
          << batch_state_->next_flush_event.fence_value;

  absl::InlinedVector<uint8_t, 16> value_copy(value.size());
  std::copy(value.begin(), value.end(), value_copy.begin());

  batch_state_->WriteBatch().emplace_back(
      [=, value_copy = std::move(value_copy)](DmlCommandList& command_list) {
        command_list.FillBufferWithPattern(dst, dst_offset, dst_size_in_bytes,
                                           value_copy);
      });

  return batch_state_->next_flush_event;
}

DmlGpuEvent DmlExecutionContext::InitializeOperator(
    IDMLOperatorInitializer* initializer,
    Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
    ID3D12DescriptorHeap* descriptor_heap) {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  VLOG(3) << "DML EC: Batch InitializeOperator with fv = "
          << batch_state_->next_flush_event.fence_value;

  batch_state_->WriteBatch().emplace_back(
      [=,
       binding_table = std::move(binding_table)](DmlCommandList& command_list) {
        command_list.InitializeOperator(initializer, binding_table.Get(),
                                        descriptor_heap);
      });

  batch_state_->new_function_enqueued.notify_all();

  return batch_state_->next_flush_event;
}

DmlGpuEvent DmlExecutionContext::ExecuteOperator(
    IDMLCompiledOperator* op,
    Microsoft::WRL::ComPtr<IDMLBindingTable>&& binding_table,
    ID3D12DescriptorHeap* descriptor_heap) {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  VLOG(3) << "DML EC: Batch ExecuteOperator with fv = "
          << batch_state_->next_flush_event.fence_value;

  batch_state_->WriteBatch().emplace_back(
      [=,
       binding_table = std::move(binding_table)](DmlCommandList& command_list) {
        command_list.ExecuteOperator(op, binding_table.Get(), descriptor_heap);
      });

  batch_state_->new_function_enqueued.notify_all();

  return batch_state_->next_flush_event;
}

DmlGpuEvent DmlExecutionContext::ResourceBarrier(
    absl::Span<const D3D12_RESOURCE_BARRIER> barriers) {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  VLOG(3) << "DML EC: Batch ResourceBarrier with fv = "
          << batch_state_->next_flush_event.fence_value;

  // The caller may not keep the barriers referenced by the span alive for
  // longer than this function call, so make a copy and transfer ownership to
  // the lambda.
  absl::InlinedVector<D3D12_RESOURCE_BARRIER, 4> barriers_copy(barriers.begin(),
                                                               barriers.end());
  batch_state_->WriteBatch().emplace_back(
      [=, barriers = std::move(barriers_copy)](DmlCommandList& command_list) {
        command_list.ResourceBarrier(barriers);
      });

  batch_state_->new_function_enqueued.notify_all();

  return batch_state_->next_flush_event;
}

DmlGpuEvent DmlExecutionContext::UavBarrier() {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  VLOG(3) << "DML EC: Batch UavBarrier with fv = "
          << batch_state_->next_flush_event.fence_value;

  batch_state_->WriteBatch().emplace_back(
      [=](DmlCommandList& command_list) { command_list.UavBarrier(); });

  batch_state_->new_function_enqueued.notify_all();

  return batch_state_->next_flush_event;
}

StatusOr<DmlGpuEvent> DmlExecutionContext::Flush() {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  auto event = batch_state_->next_flush_event;
  if (batch_state_->WriteBatch().empty()) {
    --event.fence_value;
  }

  VLOG(3) << "DML EC: flush requested with fv = " << event.fence_value;

  batch_state_->flush_requested = true;
  batch_state_->new_function_enqueued.notify_all();
  return event;
}

Status DmlExecutionContext::GetCommandRecorderStatus() const {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  return batch_state_->status;
}

DmlGpuEvent DmlExecutionContext::GetCurrentCompletionEvent() {
  std::unique_lock<std::mutex> lock(batch_state_->mutex);
  auto event = batch_state_->next_flush_event;
  if (batch_state_->WriteBatch().empty()) {
    --event.fence_value;
  }
  return event;
}

D3D12_COMMAND_LIST_TYPE DmlExecutionContext::GetCommandListTypeForQueue()
    const {
  // No need to acquire the lock since the queue type is immutable once the
  // queue is constructed.
  return dml_command_queue_->GetType();
}

/*static*/ void DmlExecutionContext::RecordCommands(
    uint32_t thread_id, ExecutionState* exec_state) {
  auto& command_list = exec_state->command_lists[thread_id];
  auto start_command = exec_state->command_starts[thread_id];
  auto command_count = exec_state->command_counts[thread_id];
  auto commands = exec_state->commands.subspan(start_command, command_count);

  VLOG(3) << "DML EC: RecordCommands for tid = " << thread_id
          << " using fv = " << exec_state->batch_completion_event.fence_value
          << "on CL addr " << &command_list << "; start=" << start_command
          << " count=" << command_count;

  command_list.Open(exec_state->batch_completion_event);
  for (auto& command : commands) {
    command(command_list);
  }
  exec_state->command_status[thread_id] = command_list.Close();
  exec_state->command_counts[thread_id] = 0;
}

/*static*/ void DmlExecutionContext::MainExecutionThreadProc(
    std::shared_ptr<BatchState> state,
    std::shared_ptr<ExecutionState> exec_state,
    DmlCommandQueue* dml_command_queue, uint32_t batch_flush_size,
    uint32_t batch_flush_time_us) {
  auto last_flush_time = std::chrono::high_resolution_clock::now();

  while (true) {
    std::chrono::duration<double> elapsed =
        std::chrono::high_resolution_clock::now() - last_flush_time;
    auto elapsed_us = elapsed.count() * 1e6;

    std::unique_lock<std::mutex> lock(state->mutex);
    if (state->exit_requested) {
      break;
    }

    auto& batch = state->WriteBatch();

    if (batch.empty()) {
      // Wait for new work to be batched.
      state->new_function_enqueued.wait(lock);
      continue;
    }

    // Check if it's time to swap the write/execute batches and flush work to
    // the GPU: this occurs if a flush is explicitly requested, the batch has
    // reached a certain size, or enough time has elapsed since the last flush.
    // The goal here is to balance feeding the GPU work while the CPU is
    // processing more commands and avoiding many small packets.
    bool flush = false;
    if (state->flush_requested || batch.size() >= batch_flush_size ||
        elapsed_us >= batch_flush_time_us) {
      state->write_batch_index = (state->write_batch_index + 1) % 2;
      flush = true;
      exec_state->batch_completion_event = state->next_flush_event;
      ++state->next_flush_event.fence_value;
    }
    state->flush_requested = false;

    // Unlock to allow kernels to resume writing to the new write batch.
    lock.unlock();

    // Invoke the batched functions and submit the work to the GPU.
    if (flush) {
      // If the batch is smaller than the number of available execution threads
      // don't bother parallelizing the recording; just record everything on the
      // current execution thread.
      const uint32_t available_threads =
          static_cast<uint32_t>(exec_state->command_lists.size());
      assert(available_threads >= 1);
      uint32_t threads_used =
          batch.size() < available_threads ? 1 : available_threads;

      VLOG(3) << "DML EC: Flush batch using fv = "
              << exec_state->batch_completion_event.fence_value;

      // Distribute functions evenly to threads, with any remainder going to
      // lower index threads first. With N threads, the first N-1 threads are
      // "helper" threads that merely record into their own command list. The
      // last thread launches the helper threads, records its own commands, and
      // then joins the helper threads (i.e. the last thread is the one running
      // this function).
      uint32_t commands_per_thread = batch.size() / threads_used;
      uint32_t remainder = batch.size() % threads_used;
      uint32_t start_index = 0;

      // Divide the command batch into subspans.
      exec_state->commands = absl::Span<Command>{batch.data(), batch.size()};
      std::unique_lock<std::mutex> exec_state_lock(exec_state->mutex);
      for (uint32_t thread_id = 0; thread_id < threads_used; thread_id++) {
        uint32_t command_count =
            commands_per_thread + (thread_id < remainder ? 1 : 0);
        exec_state->command_starts[thread_id] = start_index;
        exec_state->command_counts[thread_id] = command_count;
        start_index += command_count;
      }
      exec_state_lock.unlock();

      // Wake the helper threads so they start recording.
      if (threads_used > 1) {
        exec_state->commands_added.notify_all();
      }

      // Record commands in this thread as well.
      RecordCommands(0, exec_state.get());

      // Wait for all threads to finish recording.
      // TODO busy wait for now; maybe replace with cond var work_completed in a
      // loop until all done.
      bool work_pending = false;
      do {
        work_pending = false;
        for (uint32_t thread_id = 1; thread_id < threads_used; thread_id++) {
          if (exec_state->command_counts[thread_id] != 0) {
            work_pending = true;
            break;
          }
        }
      } while (work_pending);

      // Gather recorded commands lists and check their status.
      absl::InlinedVector<ID3D12CommandList*, default_num_exec_threads>
          d3d_command_lists;
      for (uint32_t thread_id = 0; thread_id < threads_used; thread_id++) {
        // Bail if any errors occurred while recording commands.
        if (!exec_state->command_status[thread_id].ok()) {
          VLOG(3) << "DML EC: BAD STATUS!";
          lock.lock();
          state->status = exec_state->command_status[thread_id];
          lock.unlock();
          break;
        }

        d3d_command_lists.push_back(exec_state->command_lists[thread_id].Get());
      }

      // Execute command lists and clear the batch.
      dml_command_queue->ExecuteCommandLists(
          {d3d_command_lists.data(), d3d_command_lists.size()});
      batch.clear();

      last_flush_time = std::chrono::high_resolution_clock::now();
    }
  }
}

/*static*/ void DmlExecutionContext::SecondaryExecutionThreadProc(
    uint32_t thread_id, std::shared_ptr<ExecutionState> exec_state) {
  while (true) {
    std::unique_lock<std::mutex> lock(exec_state->mutex);
    if (exec_state->command_counts[thread_id] == 0) {
      exec_state->commands_added.wait(lock);
      continue;
    }
    lock.unlock();

    RecordCommands(thread_id, exec_state.get());
  }
}
}  // namespace tensorflow
