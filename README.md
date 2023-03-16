**vk_memory_decompression**

This example shows how to use the VK_NV_memory_decompression Vulkan memory decompresion extension to compress and decompress with
NVIDIA GDeflate using the NVIDIA libdeflate fork.

~~~~ batch
vk_memory_decompression.exe OPTION
     -c <level> : compress input file to output file at level 1..12
     -d         : (optional) decompress on CPU
     -i <file>  : input file
     -o <file>  : output file
~~~~
