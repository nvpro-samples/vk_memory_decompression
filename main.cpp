/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <vulkan/vulkan.hpp>

#include "nvh/alignment.hpp"
#include "nvp/nvpsystem.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/inputparser.h"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/vulkanhppsupport.hpp"


#include "libdeflate/libdeflate.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

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

//--------------------------------------------------------------------------------------------------
// A trivial paged compressed stream
//
// Format:
//
// [magic number:32]
// [(page 0 compressed size - 1):32]
// [page 0 compressed bits]
// ...
// [(page N compressed size - 1):32]
// [page N compressed bits]
// [end tag:32]
// [page N uncompressed size:32]
// [crc32:32]
//
// Pages 0 to N-1 all have uncompressed size equal to kGDeflatePageSize. The remainder
// tail page N uncompressed size is stored in the footer of the stream. Compressed sizes of each
// page is stored in the PageHeader. A special tag kEndOfPagesTag signals the end-of-pages
// condition, and is immedeately followed by stream footer data.
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

  explicit CompressedPageStream(const std::string& filename)
      : BaseClass(filename.c_str(), BaseClass::binary | (IsInput ? BaseClass::in : BaseClass::out))
  {
    if(BaseClass::is_open() && BaseClass::good())
    {
      if constexpr(IsInput)
      {
        char magic[5] = {0};
        this->read(magic, 4);

        if(0 != strcmp(magic, kMagic))
        {
          LOGE("File %s is not a GDEFLATE sample page stream.\n", filename.c_str());
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
static bool Compress(const std::string& outFile, const std::string& inFile, uint32_t level)
{
  std::ifstream fin(inFile, std::ifstream::binary | std::ifstream::in);

  if(!fin.is_open() || !fin.good())
  {
    LOGE("Unable to open: %s\n", inFile.c_str());
    return false;
  }

  OutputCompressedPageStream fout(outFile);

  if(!fout.is_open() || !fout.good())
  {
    LOGE("Unable to create file: %s\n", outFile.c_str());
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

  LOGI("Compressing %s", inFile.c_str());

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
static bool DecompressCPU(const std::string& outFile, const std::string& inFile)
{
  InputCompressedPageStream fin(inFile);

  if(!fin.is_open() || !fin.good())
  {
    LOGE("Unable to open file: %s\n", inFile.c_str());
    return false;
  }

  std::ofstream fout(outFile, std::ofstream::binary | std::ofstream::out);

  if(!fout.is_open() || !fout.good())
  {
    LOGE("Unable to create file: %s\n", outFile.c_str());
    return false;
  }

  auto d = libdeflate_alloc_gdeflate_decompressor();

  if(d == nullptr)
  {
    LOGE("Unable to allocate GDEFLATE decompressor.\n");
    return false;
  }

  LOGI("Decompressing %s on CPU", inFile.c_str());

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
static bool DecompressGPU(const std::string& outFile, const std::string& inFile, nvvk::Context& ctx)
{
  InputCompressedPageStream fin(inFile);

  if(!fin.is_open() || !fin.good())
  {
    LOGE("Unable to open file: %s\n", inFile.c_str());
    return false;
  }

  std::ofstream fout(outFile, std::ofstream::binary | std::ofstream::out);

  if(!fout.is_open() || !fout.good())
  {
    LOGE("Unable to create file: %s\n", outFile.c_str());
    return false;
  }

  LOGI("Decompressing %s\n", inFile.c_str());

  // Allocate intermediate buffers
  nvvkpp::ResourceAllocatorDedicated alloc;
  alloc.init(ctx.m_device, ctx.m_physicalDevice);

  // Buffer for transfer
  nvvk::Buffer transferBuffer =
      alloc.createBuffer(kIntermediateBufferSize,
                         vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferSrc
                             | vk::BufferUsageFlagBits::eTransferDst,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached);

  // Buffer for compressed data
  nvvk::Buffer srcBuffer = alloc.createBuffer(kIntermediateBufferSize,
                                              vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst
                                                  | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                              vk::MemoryPropertyFlagBits::eDeviceLocal);

  VkDeviceAddress srcAddr = nvvk::getBufferDeviceAddress(ctx.m_device, srcBuffer.buffer);

  // Buffer for uncompressed data
  nvvk::Buffer dstBuffer = alloc.createBuffer(kIntermediateBufferSize,
                                              vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferSrc
                                                  | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                              vk::MemoryPropertyFlagBits::eDeviceLocal);

  VkDeviceAddress dstAddr = nvvk::getBufferDeviceAddress(ctx.m_device, dstBuffer.buffer);

  // Indirect param buffer
  nvvk::Buffer paramBuffer =
      alloc.createBuffer(sizeof(VkDeviceSize) + sizeof(VkDecompressMemoryRegionNV) * kMaxPagesPerBatch,
                         vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eIndirectBuffer
                             | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached);

  VkDeviceAddress paramAddr = nvvk::getBufferDeviceAddress(ctx.m_device, paramBuffer.buffer);

  // Command Buffer Pool Generator
  nvvk::CommandPool genCmdBuf(ctx.m_device, ctx.m_queueGCT.queueIndex);

  // Map transfer buffer
  char* transferPtr = static_cast<char*>(alloc.map(transferBuffer));

  // Map indirect parameters buffer
  auto pageCountPtr = static_cast<VkDeviceSize*>(alloc.map(paramBuffer));

  // Get pointer to decompression regions
  auto regionsPtr = reinterpret_cast<VkDecompressMemoryRegionNV*>(pageCountPtr + 1);

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
      auto region                 = regionsPtr + *pageCountPtr;
      region->srcAddress          = srcAddr + compressedBytesThisBatch;
      region->dstAddress          = dstAddr + page * kGDeflatePageSize;
      region->compressedSize      = lastPageCompressedSize;
      region->decompressedSize    = kGDeflatePageSize;
      region->decompressionMethod = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_NV;

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
      auto tbmi = alloc.getMemoryAllocator()->getMemoryInfo(transferBuffer.memHandle);
      auto pbmi = alloc.getMemoryAllocator()->getMemoryInfo(paramBuffer.memHandle);

      // vkFlushMappedMemoryRanges requires sizes to be a multiple of nonCoherentAtomSize:
      const VkDeviceSize atomSize = ctx.m_physicalInfo.properties10.limits.nonCoherentAtomSize;

      vk::MappedMemoryRange flushRanges[2] = {
          vk::MappedMemoryRange(tbmi.memory, 0, nvh::align_up(compressedBytesThisBatch, atomSize)),
          vk::MappedMemoryRange(pbmi.memory, 0,
                                nvh::align_up(sizeof(*pageCountPtr) + *pageCountPtr * sizeof(VkDecompressMemoryRegionNV), atomSize))};

      vkFlushMappedMemoryRanges(ctx.m_device, std::extent_v<decltype(flushRanges)>,
                                reinterpret_cast<const VkMappedMemoryRange*>(flushRanges));
    }

    vk::CommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();

    // Copy compressed data from transfer buffer to source buffer in GPU-local memory
    // NOTE: direct decoding from CPU-local GPU-visible memory may not be performant!
    auto region = vk::BufferCopy().setSrcOffset(0).setSize(compressedBytesThisBatch);
    cmdBuf.copyBuffer(transferBuffer.buffer, srcBuffer.buffer, 1, &region);

    auto barrier = vk::BufferMemoryBarrier()
                       .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                       .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
                       .setSrcQueueFamilyIndex(ctx.m_queueGCT.queueIndex)
                       .setDstQueueFamilyIndex(ctx.m_queueGCT.queueIndex)
                       .setBuffer(srcBuffer.buffer)
                       .setOffset(0)
                       .setSize(VK_WHOLE_SIZE);

    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &barrier, 0, nullptr);

    // Dispatch decompression command for this batch
#ifdef VK_NV_memory_decompression
    cmdBuf.decompressMemoryIndirectCountNV(paramAddr + sizeof(*pageCountPtr), paramAddr, sizeof(VkDecompressMemoryRegionNV));
#else
    vkCmdDecompressMemoryIndirectCountNV(cmdBuf, paramAddr + sizeof(*pageCountPtr), paramAddr, sizeof(VkDecompressMemoryRegionNV));
#endif

    barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
        .setDstAccessMask(vk::AccessFlagBits::eTransferRead)
        .setBuffer(dstBuffer.buffer);

    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &barrier, 0, nullptr);

    barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderRead)
        .setDstAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setBuffer(srcBuffer.buffer);

    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &barrier, 0, nullptr);

    // Download decoded data from the destination GPU-local buffer to transfer buffer
    region.setSize(uncompressedBytesThisBatch);
    cmdBuf.copyBuffer(dstBuffer.buffer, transferBuffer.buffer, 1, &region);

    // Submit and wait for result.
    // NOTE: ideally decompression stage should be pipelined with the IO and upload stages.
    genCmdBuf.submitAndWait(cmdBuf);  // Destroys cmdBuf

    // Update rolling crc32
    crc32 = libdeflate_crc32(crc32, transferPtr, uncompressedBytesThisBatch);

    // Write decompressed data
    fout.write(transferPtr, uncompressedBytesThisBatch);
  }

  // Verify crc32
  if(expectedCrc32 != crc32)
  {
    LOGE("CRC32 failed (got %x, expected %x)!\n", crc32, expectedCrc32);
    return false;
  }

  LOGI("Done!\n");

  // Clean up
  alloc.destroy(srcBuffer);
  alloc.destroy(dstBuffer);
  alloc.destroy(transferBuffer);
  alloc.destroy(paramBuffer);
  alloc.deinit();

  return true;
}

//--------------------------------------------------------------------------------------------------
// Entry of the example, see OPTIONS for the arguments
//
int main(int argc, char** argv)
{
  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  InputParser parser(argc, argv);

  const std::string in  = parser.getString("-i");
  const std::string out = parser.getString("-o");

  if(parser.exist("-h") || in.empty() || out.empty())
  {
    LOGE("\n vk_memory_decompression.exe OPTIONS\n");
    LOGE("     -c <level> : compress input file to output file at level 1..12\n");
    LOGE("     -d         : (optional) decompress on CPU\n");
    LOGE("     -i <file>  : input file\n");
    LOGE("     -o <file>  : output file\n");
    return 1;
  }

  if(parser.exist("-c"))
  {
    // Do compression if requested
    const uint32_t level = parser.getInt("-c", 8);
    return Compress(out, in, level) ? 0 : -1;
  }

  if(parser.exist("-d"))
  {
    // Do CPU decompression if requested
    return DecompressCPU(out, in) ? 0 : -1;
  }

  // Do GPU decompression by default

  // Creating the Vulkan instance and device
  nvvk::Context           vkctx;
  nvvk::ContextCreateInfo vkctxInfo{};

  // Using Vulkan 1.2+ for native buffer device address support
  vkctxInfo.setVersion(1, 2);

  // Request decompression extension
  vkctxInfo.addDeviceExtension(VK_NV_MEMORY_DECOMPRESSION_EXTENSION_NAME);

  if(!vkctx.init(vkctxInfo))
  {
    LOGE("Vulkan context initialization failed!");
    return -1;
  }

  // Initialize Vulkan function pointers
#if VK_HEADER_VERSION >= 304
  vk::detail::DynamicLoader dl;
#else
  vk::DynamicLoader dl;
#endif
  auto              vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");

  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkctx.m_instance, vkGetInstanceProcAddr, vkctx.m_device);

#ifndef VK_NV_memory_decompression
  // Initialize decompression extension function pointer
  vkCmdDecompressMemoryIndirectCountNV =
      PFN_vkCmdDecompressMemoryIndirectCountNV(vkGetInstanceProcAddr(vkctx.m_instance, "vkCmdDecompressMemoryIndirectCountNV"));
#endif

  // Printing which GPU we are using
  vk::PhysicalDevice pd(vkctx.m_physicalDevice);
  LOGI("Using GPU: %s\n", pd.getProperties().deviceName.data());

  const bool result = DecompressGPU(out, in, vkctx);

  vkctx.deinit();

  return result ? 0 : -1;
}
