/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//--------------------------------------------------------------------------------------------------
// Vulkan memory decompression example
//

#define VMA_IMPLEMENTATION

#include "libdeflate/libdeflate.h"

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/context.hpp>
#include <nvvk/resource_allocator.hpp>
#include <volk/volk.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <vector>

//--------------------------------------------------------------------------------------------------
// Declare extension if it is not in Vulkan SDK headers
#ifndef VK_NV_memory_decompression
#define VK_NV_MEMORY_DECOMPRESSION_EXTENSION_NAME "VK_NV_memory_decompression"

typedef VkFlags64 VkMemoryDecompressionMethodFlagsNV;

typedef void(VKAPI_PTR* PFN_vkCmdDecompressMemoryIndirectCountNV)(VkCommandBuffer commandBuffer,
                                                                  VkDeviceAddress indirectCommandsAddress,
                                                                  VkDeviceAddress indirectCommandsCountAddress,
                                                                  uint32_t        stride);

typedef enum VkMemoryDecompressionMethodFlagBitsNV
{
  VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_NV = 1,
} VkMemoryDecompressionMethodFlagBitsNV;

typedef struct VkDecompressMemoryRegionNV
{
  VkDeviceAddress                    srcAddress;
  VkDeviceAddress                    dstAddress;
  VkDeviceSize                       compressedSize;
  VkDeviceSize                       decompressedSize;
  VkMemoryDecompressionMethodFlagsNV decompressionMethod;
} VkDecompressMemoryRegionNV;

static PFN_vkCmdDecompressMemoryIndirectCountNV vkCmdDecompressMemoryIndirectCountNV = nullptr;
#endif

//--------------------------------------------------------------------------------------------------
// GDeflate page size
static constexpr size_t kGDeflatePageSize = 65536;
// The maximum number of GDeflate pages to process per batch
static constexpr size_t kMaxPagesPerBatch = 512;
// The size of intermediate buffer storage required
static constexpr size_t kIntermediateBufferSize = kMaxPagesPerBatch * kGDeflatePageSize;
// API version required by this sample
static constexpr uint32_t kVulkanApiVersion = VK_API_VERSION_1_4;

//--------------------------------------------------------------------------------------------------
// A trivial paged compressed stream
//
// Format:
//
// [magic number:32]
// [(page 0 compressed size - 1):32]
// [page 0 compressed bits]
// ...
// [(page N-1 compressed size - 1):32]
// [page N-1 compressed bits]
// [end tag:32]
// [page N uncompressed size:32]
// [crc32:32]
//
// Pages 0 to N-1 all have uncompressed size equal to kGDeflatePageSize. The remainder
// tail page N uncompressed size is stored in the footer of the stream. Compressed sizes of each
// page is stored in the PageHeader. A special tag kEndOfPagesTag signals the end-of-pages
// condition, and is immediately followed by stream footer data.
//
template <bool IsInput>
class CompressedPageStream : public std::conditional_t<IsInput, std::ifstream, std::ofstream>
{
  typedef std::conditional_t<IsInput, std::ifstream, std::ofstream> BaseClass;
  // Note: This declaration should be read as "kMagic is a static constexpr pointer to const char":
  static constexpr const char* kMagic = "GDEF";

public:
  // Page header structure preceding each compressed page data
  struct PageHeader
  {
    uint32_t compressedSize;
  };

  // Stream footer
  struct Footer
  {
    uint32_t tailPageUncompressedSize;
    uint32_t crc32;
  };

  // Special end-of-pages tag
  static constexpr uint32_t kEndOfPagesTag = 1;
  inline static PageHeader  endOfPagesTag{kEndOfPagesTag};

  explicit CompressedPageStream(const std::filesystem::path& path)
      : BaseClass(path, BaseClass::binary | (IsInput ? BaseClass::in : BaseClass::out))
  {
    if(BaseClass::is_open() && BaseClass::good())
    {
      if constexpr(IsInput)
      {
        char magic[5] = {0};
        this->read(magic, 4);

        if(0 != strcmp(magic, kMagic))
        {
          LOGE("File %s is not a GDEFLATE sample page stream.\n", nvutils::utf8FromPath(path).c_str());
          this->setstate(std::ios_base::badbit);
        }
      }
      else
      {
        this->write(kMagic, 4);
      }
    }
  }

  CompressedPageStream& operator<<(const libdeflate_gdeflate_out_page& page)
  {
    if constexpr(!IsInput)
    {
      const PageHeader pageHeader{static_cast<uint32_t>(page.nbytes)};
      this->write(reinterpret_cast<const char*>(&pageHeader), sizeof(pageHeader));
      this->write(static_cast<char*>(page.data), page.nbytes);
    }

    return *this;
  }

  CompressedPageStream& operator<<(libdeflate_gdeflate_in_page& page)
  {
    if constexpr(IsInput)
    {
      if(page.data == nullptr || page.nbytes == 0)
        return *this;

      PageHeader pageHeader;
      if(!readChecked(&pageHeader, sizeof(pageHeader)))
        return *this;

      if(pageHeader.compressedSize != kEndOfPagesTag)
      {
        if(pageHeader.compressedSize > page.nbytes)
        {
          LOGE("Page read buffer too small for a page size %d.\n", pageHeader.compressedSize);
          throw("Page read buffer too small");
        }

        page.nbytes = pageHeader.compressedSize;
        readChecked(const_cast<void*>(page.data), pageHeader.compressedSize);
      }
      else
      {
        // Report end of pages. Stream is still good for a footer
        page.nbytes = kEndOfPagesTag;
      }
    }

    return *this;
  }

  CompressedPageStream& operator<<(PageHeader& pageHeader)
  {
    if constexpr(IsInput)
    {
      readChecked(&pageHeader, sizeof(pageHeader));
    }
    else
    {
      this->write(reinterpret_cast<char*>(&pageHeader), sizeof(pageHeader));
    }

    return *this;
  }

  CompressedPageStream& operator<<(Footer& footer)
  {
    if constexpr(IsInput)
    {
      readChecked(&footer, sizeof(footer));
    }
    else
    {
      this->write(reinterpret_cast<char*>(&footer), sizeof(footer));
    }

    return *this;
  }

private:
  template <typename T>
  bool readChecked(T* out, size_t size)
  {
    if constexpr(IsInput)
    {
      const size_t pos = this->tellg();

      this->read(reinterpret_cast<char*>(out), size);

      if(this->gcount() != size)
      {
        LOGE("Corrupted GDEFLATE sample page stream at %zd.\n", pos);
        this->setstate(std::ios_base::failbit);
        return false;
      }

      return true;
    }

    return false;
  }
};

//--------------------------------------------------------------------------------------------------
typedef CompressedPageStream<false> OutputCompressedPageStream;
typedef CompressedPageStream<true>  InputCompressedPageStream;

//--------------------------------------------------------------------------------------------------
// Compress on CPU the 'inFile' to 'outFile' at compression level 'level'.
//
// This is a batched compressor that will be compressing up to kIntermediateBufferSize bytes
// of data per batch.
//
static bool Compress(const std::filesystem::path& outFile, const std::filesystem::path& inFile, uint32_t level)
{
  std::ifstream fin(inFile, std::ifstream::binary | std::ifstream::in);

  if(!fin.is_open() || !fin.good())
  {
    LOGE("Unable to open: %s\n", nvutils::utf8FromPath(inFile).c_str());
    return false;
  }

  OutputCompressedPageStream fout(outFile);

  if(!fout.is_open() || !fout.good())
  {
    LOGE("Unable to create file: %s\n", nvutils::utf8FromPath(outFile).c_str());
    return false;
  }

  level  = level == 0 ? 1 : level;
  level  = level > 12 ? 12 : level;
  auto c = libdeflate_alloc_gdeflate_compressor(level);

  if(c == nullptr)
  {
    LOGE("Unable to allocate GDEFLATE compressor.\n");
    return false;
  }

  size_t   uncompressedSize = 0;
  uint32_t crc32            = 0;

  std::vector<char>                         uncompressed(kIntermediateBufferSize);
  std::vector<char>                         compressed(kIntermediateBufferSize);
  std::vector<libdeflate_gdeflate_out_page> pages(kIntermediateBufferSize / kGDeflatePageSize + 1);

  LOGI("Compressing %s", nvutils::utf8FromPath(inFile).c_str());

  while(true)
  {
    fin.read(uncompressed.data(), uncompressed.size());

    const size_t bytesRead = fin.gcount();

    if(bytesRead == 0)
      break;

    // Calculate the number of GDEFLATE pages and the amount of intermediate
    // memory required
    size_t npages;
    size_t compBound     = libdeflate_gdeflate_compress_bound(nullptr, bytesRead, &npages);
    size_t pageCompBound = compBound / npages;

    // Make sure compressed data fits.
    // NOTE: compress bound value can be larger than the uncompressed data size!
    compressed.resize(compBound);
    pages.resize(npages);

    // Initialize output page table
    uint32_t pageIdx = 0;
    for(auto& page : pages)
    {
      page.data   = compressed.data() + pageIdx * pageCompBound;
      page.nbytes = pageCompBound;
      ++pageIdx;
    }

    // Compress pages
    if(libdeflate_gdeflate_compress(c, uncompressed.data(), bytesRead, pages.data(), npages))
    {
      // Gather and write compressed pages to output stream
      for(auto& page : pages)
      {
        fout << page;
      }

      // Update rolling crc32
      crc32 = libdeflate_crc32(crc32, uncompressed.data(), bytesRead);
    }
    else
    {
      LOGE(" failed!");
      return false;
    }

    uncompressedSize += bytesRead;

    LOGI(".");
  }

  // Write end-of-pages tag
  fout << OutputCompressedPageStream::endOfPagesTag;

  // Write stream footer
  OutputCompressedPageStream::Footer footer{0};

  if(uncompressedSize > 0)
  {
    footer.tailPageUncompressedSize = uncompressedSize % kGDeflatePageSize > 0 ? uncompressedSize % kGDeflatePageSize : kGDeflatePageSize;
    footer.crc32 = crc32;
  }

  fout << footer;

  libdeflate_free_gdeflate_compressor(c);

  const size_t compressedSize = fout.tellp();

  LOGI(" done!\n%zu bytes -> %zu bytes (ratio: %0.2f:1)\n", uncompressedSize, compressedSize,
       static_cast<float>(uncompressedSize) / compressedSize);

  return true;
}

//--------------------------------------------------------------------------------------------------
// Decompress on CPU 'inFile' to 'outFile'.
//
// A trivial non-batched CPU decompressor that decodes each GDeflate page separately.
//
static bool DecompressCPU(const std::filesystem::path& outFile, const std::filesystem::path& inFile)
{
  InputCompressedPageStream fin(inFile);

  if(!fin.is_open() || !fin.good())
  {
    LOGE("Unable to open file: %s\n", nvutils::utf8FromPath(inFile).c_str());
    return false;
  }

  std::ofstream fout(outFile, std::ofstream::binary | std::ofstream::out);

  if(!fout.is_open() || !fout.good())
  {
    LOGE("Unable to create file: %s\n", nvutils::utf8FromPath(outFile).c_str());
    return false;
  }

  auto d = libdeflate_alloc_gdeflate_decompressor();

  if(d == nullptr)
  {
    LOGE("Unable to allocate GDEFLATE decompressor.\n");
    return false;
  }

  LOGI("CPU decompressing %s\n", nvutils::utf8FromPath(inFile).c_str());

  std::vector<char> in(kGDeflatePageSize);
  std::vector<char> out(kGDeflatePageSize);
  uint32_t          crc32 = 0;

  while(true)
  {
    // Read a compressed page header
    InputCompressedPageStream::PageHeader pageHeader;
    fin << pageHeader;

    // Early out in case of the end of pages
    if(pageHeader.compressedSize == InputCompressedPageStream::kEndOfPagesTag)
      break;

    // Make sure page data fits
    in.resize(pageHeader.compressedSize);

    // Read page data
    fin.read(in.data(), in.size());

    // Early out in case of an error
    if(!fin.good())
      break;

    // Decompress page
    libdeflate_gdeflate_in_page page{in.data(), in.size()};
    size_t                      outSize = 0;
    libdeflate_gdeflate_decompress(d, &page, 1, out.data(), out.size(), &outSize);

    // Update rolling crc32
    crc32 = libdeflate_crc32(crc32, out.data(), outSize);

    // Write uncompressed data
    fout.write(out.data(), outSize);
  }

  libdeflate_free_gdeflate_decompressor(d);

  if(!fin.good())
    return false;

  // Read footer info
  InputCompressedPageStream::Footer footer;
  fin << footer;

  // Verify crc32
  if(footer.crc32 != crc32)
  {
    LOGE(" CRC32 failed (got %x, expected %x)!\n", crc32, footer.crc32);
    return false;
  }

  LOGI(" - done!");

  return true;
}

//--------------------------------------------------------------------------------------------------
// Decompress on GPU 'inFile' to 'outFile'.
//
// A batched GPU decompressor implementation that decodes up to kIntermediateBufferSize bytes
// of data and kMaxPagesPerBatch of pages in each batch.
//
// Batching of pages for decompression purposes is a preferable way of dispatching massivly
// parallel GPU decompression. It is expected that the client application will batch together
// as many pages as the target memory and/or latency requirements allow it to.
//
static VkResult DecompressGPU(const std::filesystem::path& outFile, const std::filesystem::path& inFile, nvvk::Context& ctx)
{
  InputCompressedPageStream fin(inFile);

  if(!fin.is_open() || !fin.good())
  {
    LOGE("Unable to open file: %s\n", nvutils::utf8FromPath(inFile).c_str());
    return VK_ERROR_UNKNOWN;
  }

  std::ofstream fout(outFile, std::ofstream::binary | std::ofstream::out);

  if(!fout.is_open() || !fout.good())
  {
    LOGE("Unable to create file: %s\n", nvutils::utf8FromPath(outFile).c_str());
    return VK_ERROR_UNKNOWN;
  }

  LOGI("GPU decompressing %s\n", nvutils::utf8FromPath(inFile).c_str());

  // Allocate intermediate buffers
  nvvk::ResourceAllocator alloc;
  NVVK_FAIL_RETURN(alloc.init(VmaAllocatorCreateInfo{.physicalDevice   = ctx.getPhysicalDevice(),
                                                     .device           = ctx.getDevice(),
                                                     .instance         = ctx.getInstance(),
                                                     .vulkanApiVersion = kVulkanApiVersion}));

  // Buffer for both CPU->GPU and GPU->CPU transfer
  // TODO(nbickford): Benchmark using BAR0 data here instead of requiring a copy
  nvvk::Buffer transferBuffer;
  NVVK_FAIL_RETURN(alloc.createBuffer(
      transferBuffer, kIntermediateBufferSize, VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
      VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));

  // Buffer for compressed data
  nvvk::Buffer srcBuffer;
  NVVK_FAIL_RETURN(alloc.createBuffer(srcBuffer, kIntermediateBufferSize,
                                      VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                                          | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));

  // Buffer for uncompressed data
  nvvk::Buffer dstBuffer;
  NVVK_FAIL_RETURN(alloc.createBuffer(dstBuffer, kIntermediateBufferSize,
                                      VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                                          | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));

  // Indirect param buffer
  nvvk::Buffer       paramBuffer;
  const VkDeviceSize paramBufferSize = sizeof(VkDeviceSize) + sizeof(VkDecompressMemoryRegionNV) * kMaxPagesPerBatch;
  NVVK_FAIL_RETURN(alloc.createBuffer(
      paramBuffer, paramBufferSize, VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));

  // Command Buffer Pool Generator
  const nvvk::QueueInfo& queueGCT = ctx.getQueueInfo(0);
  VkCommandPool          cmdPool  = nvvk::createTransientCommandPool(ctx.getDevice(), queueGCT.familyIndex);

  // Map transfer buffer
  char* transferPtr = reinterpret_cast<char*>(transferBuffer.mapping);

  // Map indirect parameters buffer
  VkDeviceSize* pageCountPtr = reinterpret_cast<VkDeviceSize*>(paramBuffer.mapping);

  // Get pointer to decompression regions
  VkDecompressMemoryRegionNV* regionsPtr = reinterpret_cast<VkDecompressMemoryRegionNV*>(pageCountPtr + 1);

  // Expected crc32 to be filled from stream footer
  uint32_t expectedCrc32 = 0;
  // Crc32 to be calculated from decoded data
  uint32_t crc32 = 0;
  // Last compressed size, put in this scope for carry-over in case of buffer overflow
  size_t lastPageCompressedSize = 0;
  // The loop will run until no more pages left in stream
  bool havePages = true;

  while(havePages)
  {
    // Read compressed pages to transfer buffer and create
    // decompression parameters table for a decompression batch
    *pageCountPtr                     = 0;
    size_t compressedBytesThisBatch   = 0;
    size_t uncompressedBytesThisBatch = 0;

    // Read up to kMaxPagesPerBatch pages per decompression batch
    for(uint32_t page = 0; page < kMaxPagesPerBatch; page++)
    {
      // Read next compressed page size if we do not have a carry-over page
      // pending from previous batch
      if(page > 0 || lastPageCompressedSize == 0)
      {
        InputCompressedPageStream::PageHeader pageHeader;
        fin << pageHeader;

        if(pageHeader.compressedSize == InputCompressedPageStream::kEndOfPagesTag)
        {
          // Page stream ended - read footer
          InputCompressedPageStream::Footer footer;
          fin << footer;

          if(*pageCountPtr > 0)
          {
            // If we have a batch pending - adjust the last batched page parameters
            // using the tail page uncompressed size before dispatch
            regionsPtr[*pageCountPtr - 1].decompressedSize = footer.tailPageUncompressedSize;
            uncompressedBytesThisBatch -= kGDeflatePageSize - footer.tailPageUncompressedSize;
          }

          // Store crc32 for verification later
          expectedCrc32 = footer.crc32;

          // Tell the loop we have no more pages left in stream
          havePages = false;
          break;
        }

        lastPageCompressedSize = pageHeader.compressedSize;
      }

      // Check if compressed page fits into source buffer
      if(compressedBytesThisBatch + lastPageCompressedSize > kIntermediateBufferSize)
        break;

      // Check if uncompressed page fits into destination buffer
      if(uncompressedBytesThisBatch + kGDeflatePageSize > kIntermediateBufferSize)
        break;

      // Preapare page decompression parameters
      VkDecompressMemoryRegionNV* region = regionsPtr + *pageCountPtr;
      region->srcAddress                 = srcBuffer.address + compressedBytesThisBatch;
      region->dstAddress                 = dstBuffer.address + page * kGDeflatePageSize;
      region->compressedSize             = lastPageCompressedSize;
      region->decompressedSize           = kGDeflatePageSize;
      region->decompressionMethod        = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_NV;

      // Read compressed page bits to transfer buffer
      fin.read(transferPtr + compressedBytesThisBatch, lastPageCompressedSize);

      compressedBytesThisBatch += lastPageCompressedSize;
      uncompressedBytesThisBatch += kGDeflatePageSize;
      ++*pageCountPtr;

      lastPageCompressedSize = 0;
    }

    // No more pages to decode - bail out
    if(*pageCountPtr == 0)
      break;

    // Flush the mapped buffers
    {
      NVVK_FAIL_RETURN(alloc.autoFlushBuffer(transferBuffer, 0, compressedBytesThisBatch));
      NVVK_FAIL_RETURN(alloc.autoFlushBuffer(paramBuffer, 0, paramBufferSize));
    }

    VkCommandBuffer cmdBuf = nvvk::createSingleTimeCommands(ctx.getDevice(), cmdPool);

    // Copy compressed data from transfer buffer to source buffer in GPU-local memory
    // NOTE: direct decoding from CPU-local GPU-visible memory may not be performant!
    const VkBufferCopy cpuToGpuRegion{.size = compressedBytesThisBatch};
    vkCmdCopyBuffer(cmdBuf, transferBuffer.buffer, srcBuffer.buffer, 1, &cpuToGpuRegion);

    nvvk::cmdBufferMemoryBarrier(cmdBuf, nvvk::BufferMemoryBarrierParams{.buffer = srcBuffer.buffer,
                                                                         .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                                         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT});

    // Dispatch decompression command for this batch
    vkCmdDecompressMemoryIndirectCountNV(cmdBuf, paramBuffer.address + sizeof(*pageCountPtr), paramBuffer.address,
                                         sizeof(VkDecompressMemoryRegionNV));

    std::array<VkBufferMemoryBarrier2, 2> transferBarriers;
    transferBarriers[0] =
        nvvk::makeBufferMemoryBarrier(nvvk::BufferMemoryBarrierParams{.buffer = dstBuffer.buffer,
                                                                      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                      .dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                                      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT});
    transferBarriers[1] =
        nvvk::makeBufferMemoryBarrier(nvvk::BufferMemoryBarrierParams{.buffer = srcBuffer.buffer,
                                                                      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                                      .dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                                      .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT});
    const VkDependencyInfo dependencyInfo{.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                          .bufferMemoryBarrierCount = uint32_t(transferBarriers.size()),
                                          .pBufferMemoryBarriers    = transferBarriers.data()};
    vkCmdPipelineBarrier2(cmdBuf, &dependencyInfo);

    // Download decoded data from the destination GPU-local buffer to transfer buffer
    const VkBufferCopy gpuToCpuRegion{.size = uncompressedBytesThisBatch};
    vkCmdCopyBuffer(cmdBuf, dstBuffer.buffer, transferBuffer.buffer, 1, &gpuToCpuRegion);

    // Make sure the transfer buffer is available to the host:
    const VkBufferMemoryBarrier2 hostTransferBarrier =
        nvvk::makeBufferMemoryBarrier(nvvk::BufferMemoryBarrierParams{.buffer        = transferBuffer.buffer,
                                                                      .srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                                      .dstStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT,
                                                                      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                                      .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT});
    const VkDependencyInfo hostDependencyInfo{.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                              .bufferMemoryBarrierCount = 1,
                                              .pBufferMemoryBarriers    = &hostTransferBarrier};
    vkCmdPipelineBarrier2(cmdBuf, &hostDependencyInfo);

    // Submit and wait for result.
    // NOTE: ideally decompression stage should be pipelined with the IO and upload stages.
    // This also destroys cmdBuf.
    NVVK_FAIL_RETURN(nvvk::endSingleTimeCommands(cmdBuf, ctx.getDevice(), cmdPool, queueGCT.queue));

    // And make sure the transfer buffer is visible to the host:
    NVVK_FAIL_RETURN(alloc.autoInvalidateBuffer(transferBuffer, 0, uncompressedBytesThisBatch));

    // Update rolling crc32
    crc32 = libdeflate_crc32(crc32, transferPtr, uncompressedBytesThisBatch);

    // Write decompressed data
    fout.write(transferPtr, uncompressedBytesThisBatch);
  }

  // Verify crc32
  if(expectedCrc32 != crc32)
  {
    LOGE("CRC32 failed (got %x, expected %x)!\n", crc32, expectedCrc32);
    return VK_ERROR_UNKNOWN;
  }

  LOGI(" - done!\n");

  // Clean up
  vkDestroyCommandPool(ctx.getDevice(), cmdPool, nullptr);
  alloc.destroyBuffer(srcBuffer);
  alloc.destroyBuffer(dstBuffer);
  alloc.destroyBuffer(transferBuffer);
  alloc.destroyBuffer(paramBuffer);
  alloc.deinit();

  return VK_SUCCESS;
}

//--------------------------------------------------------------------------------------------------
// Entry of the example, see OPTIONS for the arguments
//
int main(int argc, char** argv)
{
  // Set up argument parsing
  nvutils::ParameterRegistry registry;
  std::string                in, out;
  uint32_t                   cpu_compression_level = 0;
  bool                       do_cpu_compression = false, do_cpu_decompression = false;
  registry.add({.name = "input", .help = "input file", .shortName = "i"}, &in);
  registry.add({.name = "output", .help = "output file", .shortName = "o"}, &out);
  registry.add({.name = "cpu_compress", .help = "(optional) compress input file to output file at level 1..12.", .shortName = "c"},
               &cpu_compression_level);
  registry.add({.name = "cpu_decompress", .help = "Instead of decompressing on the GPU, decompress on the CPU.", .shortName = "d"},
               &do_cpu_decompression, true);

  nvutils::ParameterParser parser(
      "vk_memory_decompression: Uses the Vulkan VK_NV_memory_decompression extension to decompress GDeflate data on the GPU.");
  parser.add(registry);
  parser.parse(argc, argv);

  if(in.empty() || out.empty())
  {
    parser.printHelp();
    return EXIT_FAILURE;
  }

  if(cpu_compression_level > 0)
  {
    // Do compression if requested
    return Compress(out, in, cpu_compression_level) ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  if(do_cpu_decompression)
  {
    // Do CPU decompression if requested
    return DecompressCPU(out, in) ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  // Do GPU decompression by default

  // Creating the Vulkan instance and device
  nvvk::Context         vkctx;
  nvvk::ContextInitInfo initInfo{.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
                                 .deviceExtensions = {nvvk::ExtensionInfo{.extensionName = VK_NV_MEMORY_DECOMPRESSION_EXTENSION_NAME}},
                                 .apiVersion = kVulkanApiVersion};

  if(VK_SUCCESS != NVVK_FAIL_REPORT(vkctx.init(initInfo)))
  {
    return EXIT_FAILURE;
  }

#ifndef VK_NV_memory_decompression
  // Initialize decompression extension function pointer
  vkCmdDecompressMemoryIndirectCountNV =
      PFN_vkCmdDecompressMemoryIndirectCountNV(vkGetInstanceProcAddr(vkctx.m_instance, "vkCmdDecompressMemoryIndirectCountNV"));
#endif

  const VkResult result = DecompressGPU(out, in, vkctx);

  vkctx.deinit();

  return (VK_SUCCESS == result) ? EXIT_SUCCESS : EXIT_FAILURE;
}
