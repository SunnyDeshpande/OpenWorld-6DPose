#!/bin/bash
# DenseFusion Quick Start Script
# This script provides quick commands to run common tasks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=============================================="
echo "      DenseFusion Quick Start Menu"
echo "=============================================="
echo ""

# Check if environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source densefusion_env/bin/activate
fi

echo "Select an option:"
echo ""
echo "  1) Verify Installation"
echo "  2) Run Evaluation (No Visualization)"
echo "  3) Run Evaluation (With Visualization - 50 images)"
echo "  4) Run Evaluation (With Visualization - 20 images)"
echo "  5) Quick Test (First 100 images only)"
echo "  6) Check GPU Status"
echo "  7) View Visualization Results"
echo "  8) Clean Outputs"
echo "  9) Show System Info"
echo "  q) Quit"
echo ""
read -p "Enter choice [1-9 or q]: " choice

case $choice in
    1)
        echo -e "${BLUE}Running verification...${NC}"
        bash verify_installation.sh
        ;;
    2)
        echo -e "${BLUE}Running full evaluation (no visualization)...${NC}"
        echo "This will take ~10-15 minutes"
        python tools/eval_linemod.py \
            --dataset_root ./datasets/linemod/Linemod_preprocessed \
            --model ./trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
            --refine_model ./trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth
        ;;
    3)
        echo -e "${BLUE}Running evaluation with visualization (50 images)...${NC}"
        python eval_linemod_with_vis.py \
            --dataset_root ./datasets/linemod/Linemod_preprocessed \
            --model ./trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
            --refine_model ./trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth \
            --visualize \
            --vis_count 50
        echo ""
        echo -e "${GREEN}Visualization saved to: visualization_output/${NC}"
        ;;
    4)
        echo -e "${BLUE}Running evaluation with visualization (20 images)...${NC}"
        python eval_linemod_with_vis.py \
            --dataset_root ./datasets/linemod/Linemod_preprocessed \
            --model ./trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
            --refine_model ./trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth \
            --visualize \
            --vis_count 20
        echo ""
        echo -e "${GREEN}Visualization saved to: visualization_output/${NC}"
        ;;
    5)
        echo -e "${BLUE}Quick test on first 100 images...${NC}"
        echo "This is for testing only, not full evaluation"
        python tools/eval_linemod.py \
            --dataset_root ./datasets/linemod/Linemod_preprocessed \
            --model ./trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth \
            --refine_model ./trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth \
            2>&1 | head -n 150
        ;;
    6)
        echo -e "${BLUE}Checking GPU status...${NC}"
        echo ""
        nvidia-smi
        echo ""
        python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU memory:')
    print(f'  Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB')
    print(f'  Cached: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB')
"
        ;;
    7)
        echo -e "${BLUE}Viewing visualization results...${NC}"
        if [ -d "visualization_output" ]; then
            echo ""
            echo "Visualization images in visualization_output/:"
            ls -lh visualization_output/ | head -20
            echo ""
            echo "Total images: $(ls visualization_output/*.png 2>/dev/null | wc -l)"
            echo ""
            echo "To view images:"
            echo "  eog visualization_output/*.png"
            echo "  # or use your preferred image viewer"
        else
            echo "No visualization output found."
            echo "Run option 3 or 4 first to generate visualizations."
        fi
        ;;
    8)
        echo -e "${YELLOW}Cleaning output files...${NC}"
        read -p "This will delete visualization_output/ and log files. Continue? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            rm -rf visualization_output/
            rm -f experiments/eval_result/linemod/eval_result_logs.txt
            echo -e "${GREEN}Cleaned!${NC}"
        else
            echo "Cancelled."
        fi
        ;;
    9)
        echo -e "${BLUE}System Information:${NC}"
        echo ""
        echo "Python: $(python --version)"
        echo "Working Directory: $(pwd)"
        echo "Virtual Env: $VIRTUAL_ENV"
        echo ""
        echo "Disk Usage:"
        df -h .
        echo ""
        echo "Dataset Size:"
        du -sh datasets/linemod/Linemod_preprocessed/ 2>/dev/null || echo "Dataset not found"
        echo ""
        echo "Model Files:"
        du -sh trained_checkpoints/ 2>/dev/null || echo "Models not found"
        ;;
    q|Q)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
read -p "Press Enter to continue..."
exec bash "$0"  # Restart menu
