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

#include "dml_upload_heap.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

static D3D12_HEAP_PROPERTIES UploadHeapProps() {
  return CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
}

DmlUploadHeap::DmlUploadHeap(ID3D12Device* device,
                             DmlExecutionContext* execution_context)
    : DmlPooledHeap(device, UploadHeapProps(),
                    D3D12_RESOURCE_STATE_GENERIC_READ),
      execution_context_(execution_context) {}

StatusOr<DmlGpuEvent> DmlUploadHeap::BeginUploadToGpu(
    ID3D12Resource* dst, uint64_t dst_offset, D3D12_RESOURCE_STATES dst_state,
    absl::Span<const uint8_t> src) {
  Microsoft::WRL::ComPtr<ID3D12Resource> chunk_buffer;
  uint64_t offset_in_chunk = 0;
  Allocation* allocation = nullptr;

  // Retrieve a free chunk and add an allocation entry to it. Note that this
  // method is structured in this way to avoid holding the lock over the memcpy
  // below (as it can be slow for large amounts of data)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    assert(!src.empty());
    assert(dst->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);

    InvariantChecker checker(this);

    ReclaimAllocations();

    // Allocate space from the upload heap
    Chunk* chunk = nullptr;
    TF_RETURN_IF_ERROR(Reserve(src.size(), &chunk, &offset_in_chunk));

    assert(chunk != nullptr);
    assert(offset_in_chunk + src.size() <= chunk->capacity_in_bytes);

    // Add an allocation entry to the chunk. We don't have a done_event yet
    // (because we haven't queued the copy yet) so we set it later.
    chunk->allocations.emplace_back(static_cast<uint64_t>(src.size()),
                                    offset_in_chunk);

    allocation = &chunk->allocations.back();
  }

  // Map the upload heap and copy the source data into it at the specified
  // offset
  void* upload_heap_data = nullptr;
  DML_CHECK_SUCCEEDED(chunk_buffer->Map(0, nullptr, &upload_heap_data));
  memcpy(static_cast<byte*>(upload_heap_data) + offset_in_chunk, src.data(),
         src.size());
  chunk_buffer->Unmap(0, nullptr);

  // Copy from the upload heap into the destination resource
  DmlGpuEvent done_event = execution_context_->CopyBufferRegion(
      dst, dst_offset, dst_state, chunk_buffer.Get(), offset_in_chunk,
      D3D12_RESOURCE_STATE_GENERIC_READ, src.size());

  // Fill in the done_event on the allocation. Note that although this is done
  // outside the lock, nobody will inspect `done_event` until `has_done_event`
  // is set to true.
  allocation->done_event = done_event;
  allocation->has_done_event = true;  // atomic

  return done_event;
}

}  // namespace tensorflow
