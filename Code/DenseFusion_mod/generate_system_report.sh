#!/bin/bash
# Generate a system report for sharing with others
# This creates a text file with all relevant setup information

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_FILE="system_report_$(date +%Y%m%d_%H%M%S).txt"

echo "Generating system report..."
echo "This may take a few seconds..."

{
    echo "=============================================="
    echo "     DenseFusion System Report"
    echo "     Generated: $(date)"
    echo "=============================================="
    echo ""
    
    echo "=== System Information ==="
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -a)"
    echo "Working Directory: $(pwd)"
    echo ""
    
    echo "=== Python Environment ==="
    python --version
    echo "Virtual Env: $VIRTUAL_ENV"
    echo ""
    
    echo "=== Python Packages ==="
    echo "PyTorch:"
    python -c "import torch; print('  Version:', torch.__version__); print('  CUDA Available:', torch.cuda.is_available()); print('  CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>&1
    echo ""
    echo "NumPy:"
    python -c "import numpy; print('  Version:', numpy.__version__)" 2>&1
    echo ""
    echo "SciPy:"
    python -c "import scipy; print('  Version:', scipy.__version__)" 2>&1
    echo ""
    echo "OpenCV:"
    python -c "import cv2; print('  Version:', cv2.__version__)" 2>&1
    echo ""
    
    echo "=== GPU Information ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
        echo ""
        nvidia-smi
    else
        echo "nvidia-smi not available"
    fi
    echo ""
    
    echo "=== Disk Usage ==="
    df -h .
    echo ""
    
    echo "=== DenseFusion Directory Structure ==="
    tree -L 2 -d . 2>/dev/null || find . -maxdepth 2 -type d 2>/dev/null
    echo ""
    
    echo "=== Dataset Information ==="
    if [ -d "datasets/linemod/Linemod_preprocessed" ]; then
        echo "Dataset found: datasets/linemod/Linemod_preprocessed"
        echo "Size: $(du -sh datasets/linemod/Linemod_preprocessed 2>/dev/null | cut -f1)"
        echo ""
        echo "Objects found:"
        ls -d datasets/linemod/Linemod_preprocessed/data/*/ 2>/dev/null | wc -l
        echo ""
        echo "Sample counts per object:"
        for obj in datasets/linemod/Linemod_preprocessed/data/*/; do
            obj_name=$(basename "$obj")
            count=$(ls "$obj"*-color.png 2>/dev/null | wc -l)
            echo "  Object $obj_name: $count images"
        done
    else
        echo "Dataset not found!"
    fi
    echo ""
    
    echo "=== Trained Models ==="
    if [ -d "trained_checkpoints/linemod" ]; then
        echo "Model directory found: trained_checkpoints/linemod"
        echo ""
        echo "Model files:"
        for model in trained_checkpoints/linemod/*.pth; do
            if [ -f "$model" ]; then
                size=$(ls -lh "$model" | awk '{print $5}')
                name=$(basename "$model")
                echo "  $name: $size"
            fi
        done
    else
        echo "Model directory not found!"
    fi
    echo ""
    
    echo "=== Module Import Test ==="
    python -c "
import sys
sys.path.insert(0, '.')
print('Testing imports...')
try:
    from lib.network import PoseNet, PoseRefineNet
    print('✓ Network modules: OK')
except Exception as e:
    print('✗ Network modules: FAILED -', e)

try:
    from lib.loss import Loss
    print('✓ Loss module: OK')
except Exception as e:
    print('✗ Loss module: FAILED -', e)

try:
    from lib.knn.__init__ import KNearestNeighbor
    print('✓ KNN module: OK')
except Exception as e:
    print('✗ KNN module: FAILED -', e)

try:
    from datasets.linemod.dataset import PoseDataset
    print('✓ Dataset module: OK')
except Exception as e:
    print('✗ Dataset module: FAILED -', e)
" 2>&1
    echo ""
    
    echo "=== KNN CUDA Kernel ==="
    if [ -f "lib/knn/build/knn_cuda_kernel.so" ]; then
        echo "✓ KNN CUDA kernel built: lib/knn/build/knn_cuda_kernel.so"
        ls -lh lib/knn/build/knn_cuda_kernel.so
    else
        echo "✗ KNN CUDA kernel not built (will use CPU fallback)"
    fi
    echo ""
    
    echo "=== Recent Evaluation Results ==="
    if [ -f "experiments/eval_result/linemod/eval_result_logs.txt" ]; then
        echo "Latest evaluation log found:"
        tail -20 experiments/eval_result/linemod/eval_result_logs.txt
    else
        echo "No evaluation results found yet."
    fi
    echo ""
    
    echo "=== Visualization Output ==="
    if [ -d "visualization_output" ]; then
        count=$(ls visualization_output/*.png 2>/dev/null | wc -l)
        echo "Visualization images: $count"
        if [ $count -gt 0 ]; then
            echo "Sample files:"
            ls visualization_output/*.png 2>/dev/null | head -5
        fi
    else
        echo "No visualization output yet."
    fi
    echo ""
    
    echo "=== Configuration Files ==="
    echo "Dataset config:"
    if [ -f "datasets/linemod/dataset_config/classes.txt" ]; then
        echo "  Classes file found"
        cat datasets/linemod/dataset_config/classes.txt
    fi
    echo ""
    
    echo "=============================================="
    echo "     End of System Report"
    echo "=============================================="
    
} > "$OUTPUT_FILE" 2>&1

echo ""
echo "Report generated: $OUTPUT_FILE"
echo ""
echo "You can share this file with your friends to help them set up."
echo ""
echo "View the report:"
echo "  cat $OUTPUT_FILE"
echo "  less $OUTPUT_FILE"
echo "  nano $OUTPUT_FILE"
echo ""
