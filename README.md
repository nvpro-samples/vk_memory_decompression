# vk_memory_decompression

This example shows how to use the VK_NV_memory_decompression Vulkan memory decompression extension to compress and decompress with
NVIDIA GDeflate using the NVIDIA libdeflate fork.

For example, the following commands compress a file on the CPU, then decompress it on the GPU:

```bash
./vk_memory_decompression -c -i filename -o filename.gdeflate
./vk_memory_decompression -i filename.gdeflate -o filename.decompressed
```

Options:

```output
--configfile, -cf    custom[1]: (string) - Parses provided config file. Relative filenames
                     are allowed.
--help, -h           custom: () - Prints all known parameter options registered to
                     the parser.
--input, -i          string[1]: input file
--output, -o         string[1]: output file
--cpu_compress, -c   uint32[1]: (optional) compress input file to output file at level
                     1..12.
--cpu_decompress, -d bool_trigger: Instead of decompressing on the GPU, decompress
                     on the CPU.
```
