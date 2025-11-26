#!/bin/bash
set -e

VERBOSE=""
CUTLASS_OPT=""
INT8_TC=""
MAKE_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            MAKE_ARGS="$MAKE_ARGS VERBOSE=1"
            shift
            ;;
        --cutlass-rename)
            CUTLASS_OPT=1
            MAKE_ARGS="$MAKE_ARGS CUTLASS_OPT=1"
            shift
            ;;
        --int8)
            INT8_TC=1
            MAKE_ARGS="$MAKE_ARGS INT8_TC=1"
            shift
            ;;
        --arch=*)
            MAKE_ARGS="$MAKE_ARGS FORCE_ARCH=${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Enable verbose PTXAS output (register usage, spills)"
            echo "  --cutlass        Enable experimental cutlass kernel naming optimization"
            echo "  --int8           Use INT8 tensor cores instead of FP32 cuBLAS (H100 only)"
            echo "  --arch=SM        Force specific architecture (e.g., --arch=sm_90a)"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Normal build and run (FP32 cuBLAS)"
            echo "  $0 -v                 # Build with verbose PTXAS output"
            echo "  $0 --int8             # Use INT8 tensor cores (H100)"
            echo "  $0 --int8 -v          # INT8 with verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

if [ ! -f "input.txt" ]; then
    echo "Preparing dataset..."
    if [ ! -f "wikigen/wikitext_combined.txt" ]; then
        cd wikigen
        python convert_wikitext.py
        cd ..
    fi
    ln -sf wikigen/wikitext_combined.txt input.txt
    echo "Dataset ready: $(du -h input.txt | cut -f1)"
else
    echo "Dataset already exists: $(du -h input.txt | cut -f1)"
fi

export PATH=/usr/local/cuda-12.8/bin:$PATH

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc not found in PATH; please install CUDA toolkit."
    exit 1
fi

echo ""
echo "Compiling CUDA code (full_trained_egg.cu)..."
if [ -n "$VERBOSE" ]; then
    echo "Verbose PTXAS enabled - showing register usage"
fi
if [ -n "$CUTLASS_OPT" ]; then
    echo "Cutlass kernel naming optimization enabled"
fi
if [ -n "$INT8_TC" ]; then
    echo "INT8 Tensor Core mode enabled (WGMMA on H100)"
fi

make clean
make $MAKE_ARGS

echo ""
echo "Running training..."
./egg_gpu
