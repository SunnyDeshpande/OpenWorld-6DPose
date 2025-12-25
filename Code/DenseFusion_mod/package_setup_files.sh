#!/bin/bash
# Package setup files for sharing with friends
# This creates a zip file with all necessary setup scripts and guides

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PACKAGE_NAME="densefusion_setup_$(date +%Y%m%d).zip"

echo "=============================================="
echo "   Packaging DenseFusion Setup Files"
echo "=============================================="
echo ""

# Files to include
FILES=(
    "INSTALLATION_GUIDE.md"
    "QUICK_README.md"
    "verify_installation.sh"
    "quick_start.sh"
    "generate_system_report.sh"
    "eval_linemod_with_vis.py"
    "convert_densefusion_to_bop.py"
)

# Check which files exist
echo "Checking files to package..."
FOUND_FILES=()
MISSING_FILES=()

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
        FOUND_FILES+=("$file")
    else
        echo "  ✗ $file (missing)"
        MISSING_FILES+=("$file")
    fi
done

echo ""

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "Warning: ${#MISSING_FILES[@]} file(s) missing"
fi

# Create package
echo "Creating package: $PACKAGE_NAME"

if command -v zip &> /dev/null; then
    zip -q "$PACKAGE_NAME" "${FOUND_FILES[@]}"
    echo ""
    echo "✓ Package created successfully!"
    echo ""
    echo "Package contents:"
    unzip -l "$PACKAGE_NAME"
    echo ""
    echo "Package size: $(ls -lh "$PACKAGE_NAME" | awk '{print $5}')"
    echo ""
    echo "Share this file with your friends:"
    echo "  $PACKAGE_NAME"
    echo ""
    echo "They should:"
    echo "  1. Clone DenseFusion: git clone https://github.com/j96w/DenseFusion.git"
    echo "  2. Unzip this package in the DenseFusion directory"
    echo "  3. Follow INSTALLATION_GUIDE.md"
    echo "  4. Run verify_installation.sh to check setup"
else
    # Fallback to tar if zip not available
    echo "zip not found, using tar instead..."
    tar -czf "${PACKAGE_NAME%.zip}.tar.gz" "${FOUND_FILES[@]}"
    PACKAGE_NAME="${PACKAGE_NAME%.zip}.tar.gz"
    echo ""
    echo "✓ Package created: $PACKAGE_NAME"
    echo "  (Note: Created tar.gz instead of zip)"
fi

echo ""
echo "=============================================="

# Create a README for the package
README_FILE="PACKAGE_README.txt"
cat > "$README_FILE" << 'EOF'
DenseFusion Setup Package
=========================

This package contains setup scripts and guides for DenseFusion 6D pose estimation.

Files Included:
--------------
1. INSTALLATION_GUIDE.md - Complete installation instructions
2. QUICK_README.md - Quick reference guide
3. verify_installation.sh - Verification script
4. quick_start.sh - Interactive menu
5. generate_system_report.sh - System report generator
6. eval_linemod_with_vis.py - Evaluation with visualization
7. convert_densefusion_to_bop.py - BOP format converter

Installation Steps:
------------------
1. Clone DenseFusion repository:
   git clone https://github.com/j96w/DenseFusion.git
   cd DenseFusion

2. Extract this package in the DenseFusion directory:
   unzip densefusion_setup_YYYYMMDD.zip
   # or
   tar -xzf densefusion_setup_YYYYMMDD.tar.gz

3. Follow INSTALLATION_GUIDE.md step by step

4. Verify your setup:
   bash verify_installation.sh

5. Use the interactive menu for common tasks:
   bash quick_start.sh

Quick Start:
-----------
After following the installation guide:

# Activate environment
source densefusion_env/bin/activate

# Run verification
bash verify_installation.sh

# Run evaluation
python tools/eval_linemod.py \
    --dataset_root ./datasets/linemod/Linemod_preprocessed \
    --model ./trained_checkpoints/linemod/pose_model_*.pth \
    --refine_model ./trained_checkpoints/linemod/pose_refine_model_*.pth

Important Notes:
---------------
- No sudo or conda required!
- Requires CUDA-capable GPU
- ~15GB disk space needed
- Download dataset manually from Google Drive:
  https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7

Support:
--------
- Read INSTALLATION_GUIDE.md for detailed instructions
- Run verify_installation.sh to check your setup
- Generate system report: bash generate_system_report.sh
- Original repo: https://github.com/j96w/DenseFusion

EOF

echo "Created package README: $README_FILE"
echo ""
echo "Done! Share these files with your friends:"
echo "  - $PACKAGE_NAME"
echo "  - $README_FILE"
