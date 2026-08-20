# Test runner for vk_memory_decompression.
import os, subprocess, sys

# We need a file to test compression. Let's use the executable itself!
prefix = "_install/vk_memory_decompression"
exe = prefix + (".exe" if os.name == "nt" else "")
compressed = prefix + ".gdef"


def run(args):
    print(f"\nRunning: {args}", flush=True)
    subprocess.run(args, check=True)


# Some CI systems have problems with negative exit codes, so we make sure to
# catch any errors:
try:
    # Compress the executable to GDeflate format
    run([exe, "-c", "12", "-i", exe, "-o", compressed])

    # Decompress on the GPU and CPU; check that it matches
    with open(exe, "rb") as f:
        reference = f.read()

    for use_cpu in [False, True]:
        decompressed = prefix + (".cpu" if use_cpu else ".gpu") + ".bin"
        run(
            [exe, "-i", compressed, "-o", decompressed]
            + (["-d"] if use_cpu else [])
        )

        with open(decompressed, "rb") as f:
            decompressed_data = f.read()

        if reference != decompressed_data:
            print(
                f"Decompressor didn't give the correct results for use_cpu={use_cpu}!"
            )
            sys.exit(1)

except Exception as e:
    print(e)
    sys.exit(1)
