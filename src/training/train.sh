#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' 
echo "=========================================="
echo "DiagnosticCNN Training Script"
echo "=========================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please create venv first: python3 -m venv venv"
    exit 1
fi

if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

FIVEK_DIR="data/training_datasets/FiveK"
if [ ! -d "$FIVEK_DIR" ]; then
    echo -e "${RED}Error: FiveK dataset not found at $FIVEK_DIR${NC}"
    echo "Please download FiveK dataset first"
    exit 1
fi

if ! ls "$FIVEK_DIR"/input* &> /dev/null && ! ls "$FIVEK_DIR"/source* &> /dev/null; then
    echo -e "${RED}Error: Input directory not found in $FIVEK_DIR${NC}"
    echo "Expected structure: $FIVEK_DIR/input/ (or source/)"
    exit 1
fi

if ! ls "$FIVEK_DIR"/expertC* &> /dev/null && ! ls "$FIVEK_DIR"/target* &> /dev/null; then
    echo -e "${RED}Error: Expert directory not found in $FIVEK_DIR${NC}"
    echo "Expected structure: $FIVEK_DIR/expertC/ (or target/)"
    exit 1
fi

mkdir -p models

echo -e "${YELLOW}Checking CUDA availability...${NC}"
DEVICE="cuda"
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${YELLOW}Warning: CUDA not available, using CPU${NC}"
    DEVICE="cpu"
else
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}✓ GPU detected: $GPU_NAME${NC}"
fi

BATCH_SIZE=128
EPOCHS=30
LR=0.001
NUM_WORKERS=8
OUTPUT="models/diagnostic_model.pth"

echo ""
echo "Training Configuration:"
echo "  Data:        $FIVEK_DIR"
echo "  Batch size:  $BATCH_SIZE"
echo "  Epochs:      $EPOCHS"
echo "  Learning rate: $LR"
echo "  Workers:     $NUM_WORKERS"
echo "  Device:      $DEVICE"
echo "  Output:      $OUTPUT"
echo ""

read -p "Start training? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Training cancelled"
    exit 0
fi

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo "=========================================="
echo ""

python3 src/training/train_diagnostic_model.py \
    --data "$FIVEK_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --output "$OUTPUT"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo "=========================================="
    echo ""
    echo "Model saved to: $OUTPUT"
    echo ""
    echo "Next steps:"
    echo "  1. Test the model: python predict.py <image_path>"
    echo "  2. Run the app: python app.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo -e "${RED}✗ Training failed${NC}"
    echo "=========================================="
    echo ""
    exit 1
fi
