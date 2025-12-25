#!/bin/bash
# DenseFusion Installation Verification Script
# Run this to verify your complete setup

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "   DenseFusion Installation Verification"
echo "=============================================="
echo ""

# Check counter
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# Function to print check results
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# 1. Check if virtual environment is activated
echo "Step 1: Checking Python Environment..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    check_pass "Virtual environment is activated: $VIRTUAL_ENV"
else
    check_fail "Virtual environment not activated!"
    echo "       Please run: source densefusion_env/bin/activate"
    exit 1
fi

# 2. Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
check_pass "Python version: $PYTHON_VERSION"

# 3. Check PyTorch installation
echo ""
echo "Step 2: Checking PyTorch and Dependencies..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    check_pass "PyTorch installed: $TORCH_VERSION"
    
    # Check CUDA
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        check_pass "CUDA available: Yes (GPU: $GPU_NAME)"
    else
        check_warn "CUDA not available - will run on CPU (much slower)"
    fi
else
    check_fail "PyTorch not installed!"
    echo "       Please run: pip install torch==2.0.1 torchvision==0.15.2"
fi

# 4. Check other dependencies
DEPS=("numpy" "scipy" "PIL" "yaml" "cv2" "matplotlib")
DEP_NAMES=("NumPy" "SciPy" "Pillow" "PyYAML" "OpenCV" "Matplotlib")

for i in "${!DEPS[@]}"; do
    if python -c "import ${DEPS[$i]}" 2>/dev/null; then
        VERSION=$(python -c "import ${DEPS[$i]}; print(${DEPS[$i]}.__version__)" 2>/dev/null || echo "unknown")
        check_pass "${DEP_NAMES[$i]} installed: $VERSION"
    else
        check_fail "${DEP_NAMES[$i]} not installed!"
    fi
done

# 5. Check KNN module
echo ""
echo "Step 3: Checking KNN Module..."
if python -c "from lib.knn.__init__ import KNearestNeighbor" 2>/dev/null; then
    check_pass "KNN module can be imported"
    
    # Check if CUDA kernel is built
    if [ -f "lib/knn/build/knn_cuda_kernel.so" ]; then
        check_pass "KNN CUDA kernel built: lib/knn/build/knn_cuda_kernel.so"
    else
        check_warn "KNN CUDA kernel not built (will use CPU fallback)"
    fi
else
    check_fail "KNN module cannot be imported!"
fi

# 6. Check network modules
echo ""
echo "Step 4: Checking DenseFusion Modules..."
if python -c "from lib.network import PoseNet, PoseRefineNet" 2>/dev/null; then
    check_pass "Network modules can be imported"
else
    check_fail "Network modules cannot be imported!"
fi

if python -c "from lib.loss import Loss" 2>/dev/null; then
    check_pass "Loss module can be imported"
else
    check_fail "Loss module cannot be imported!"
fi

# 7. Check dataset
echo ""
echo "Step 5: Checking Dataset..."
DATASET_ROOT="datasets/linemod/Linemod_preprocessed"

if [ -d "$DATASET_ROOT" ]; then
    check_pass "Dataset directory exists: $DATASET_ROOT"
    
    # Check data subdirectories
    OBJECT_DIRS=(01 02 04 05 06 08 09 10 11 12 13 14 15)
    OBJECTS_FOUND=0
    
    for obj in "${OBJECT_DIRS[@]}"; do
        if [ -d "$DATASET_ROOT/data/$obj" ]; then
            ((OBJECTS_FOUND++))
        fi
    done
    
    if [ $OBJECTS_FOUND -eq 13 ]; then
        check_pass "All 13 object directories found"
        
        # Check sample files in first object
        SAMPLE_FILES=$(ls "$DATASET_ROOT/data/01/"*-color.png 2>/dev/null | wc -l)
        if [ $SAMPLE_FILES -gt 0 ]; then
            check_pass "Test images found: $SAMPLE_FILES images in object 01"
        else
            check_fail "No test images found in dataset!"
        fi
    else
        check_fail "Only $OBJECTS_FOUND/13 object directories found!"
    fi
    
    # Check models
    if [ -d "$DATASET_ROOT/models" ]; then
        MODEL_COUNT=$(ls "$DATASET_ROOT/models"/*.ply 2>/dev/null | wc -l)
        if [ $MODEL_COUNT -gt 0 ]; then
            check_pass "3D models found: $MODEL_COUNT .ply files"
        else
            check_warn "No .ply model files found"
        fi
    else
        check_fail "Models directory not found!"
    fi
    
else
    check_fail "Dataset directory not found: $DATASET_ROOT"
    echo "       Please download and extract Linemod_preprocessed.zip"
fi

# 8. Check trained models
echo ""
echo "Step 6: Checking Trained Models..."
MODEL_DIR="trained_checkpoints/linemod"

if [ -d "$MODEL_DIR" ]; then
    check_pass "Model directory exists: $MODEL_DIR"
    
    # Check for .pth files
    PTH_FILES=$(ls "$MODEL_DIR"/*.pth 2>/dev/null | wc -l)
    
    if [ $PTH_FILES -gt 0 ]; then
        check_pass "Found $PTH_FILES trained model files"
        
        # Check file sizes (should be > 10MB)
        echo ""
        echo "       Model file sizes:"
        for model in "$MODEL_DIR"/*.pth; do
            if [ -f "$model" ]; then
                SIZE=$(stat -f%z "$model" 2>/dev/null || stat -c%s "$model" 2>/dev/null)
                SIZE_MB=$((SIZE / 1024 / 1024))
                FILENAME=$(basename "$model")
                
                if [ $SIZE_MB -gt 10 ]; then
                    echo "       ✓ $FILENAME: ${SIZE_MB}MB"
                else
                    echo -e "       ${RED}✗${NC} $FILENAME: ${SIZE_MB}MB (too small, likely corrupted!)"
                    ((CHECKS_FAILED++))
                fi
            fi
        done
    else
        check_fail "No .pth model files found!"
        echo "       Please download trained_checkpoints.zip from Google Drive"
    fi
else
    check_fail "Model directory not found: $MODEL_DIR"
    echo "       Please download and extract trained_checkpoints.zip"
fi

# 9. Check disk space
echo ""
echo "Step 7: Checking Disk Space..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ $AVAILABLE_SPACE -gt 5 ]; then
    check_pass "Available disk space: ${AVAILABLE_SPACE}GB"
else
    check_warn "Low disk space: ${AVAILABLE_SPACE}GB available"
fi

# 10. Test import
echo ""
echo "Step 8: Testing Complete Import..."
TEST_IMPORT=$(python -c "
import sys
sys.path.insert(0, '.')
try:
    from lib.network import PoseNet, PoseRefineNet
    from lib.loss import Loss
    from datasets.linemod.dataset import PoseDataset
    print('SUCCESS')
except Exception as e:
    print(f'FAILED: {e}')
" 2>&1)

if [[ "$TEST_IMPORT" == "SUCCESS" ]]; then
    check_pass "All modules import successfully"
else
    check_fail "Module import test failed: $TEST_IMPORT"
fi

# Summary
echo ""
echo "=============================================="
echo "               SUMMARY"
echo "=============================================="
echo -e "Checks passed:  ${GREEN}$CHECKS_PASSED${NC}"
echo -e "Checks failed:  ${RED}$CHECKS_FAILED${NC}"
echo -e "Warnings:       ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $CHECKS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo "You can now run DenseFusion evaluation:"
    echo ""
    echo "python tools/eval_linemod.py \\"
    echo "    --dataset_root ./datasets/linemod/Linemod_preprocessed \\"
    echo "    --model ./trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \\"
    echo "    --refine_model ./trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth"
    echo ""
    
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}Note: There are $WARNINGS warning(s). The system will work but may be slower.${NC}"
    fi
    
    exit 0
else
    echo -e "${RED}✗ $CHECKS_FAILED check(s) failed!${NC}"
    echo ""
    echo "Please fix the issues above before running evaluation."
    echo "See INSTALLATION_GUIDE.md for detailed troubleshooting."
    exit 1
fi
