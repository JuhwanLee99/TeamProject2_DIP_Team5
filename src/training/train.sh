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

# Use venv Python explicitly
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

FIVEK_DIR="data/training_datasets/FiveK"
if [ ! -d "$FIVEK_DIR" ]; then
    echo -e "${RED}Error: FiveK dataset not found at $FIVEK_DIR${NC}"
    echo "Please download FiveK dataset first"
    exit 1
fi

if ! ls "$FIVEK_DIR"/raw &> /dev/null && ! ls "$FIVEK_DIR"/input* &> /dev/null && ! ls "$FIVEK_DIR"/source* &> /dev/null; then
    echo -e "${RED}Error: Input directory not found in $FIVEK_DIR${NC}"
    echo "Expected structure: $FIVEK_DIR/raw/ (or input/ or source/)"
    exit 1
fi

if ! ls "$FIVEK_DIR"/c &> /dev/null && ! ls "$FIVEK_DIR"/expertC* &> /dev/null && ! ls "$FIVEK_DIR"/target* &> /dev/null; then
    echo -e "${RED}Error: Expert directory not found in $FIVEK_DIR${NC}"
    echo "Expected structure: $FIVEK_DIR/c/ (or expertC/ or target/)"
    exit 1
fi

mkdir -p models

echo -e "${YELLOW}Checking CUDA availability...${NC}"
DEVICE="cuda"
if ! "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${YELLOW}Warning: CUDA not available, using CPU${NC}"
    DEVICE="cpu"
else
    GPU_NAME=$("$PYTHON_BIN" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}✓ GPU detected: $GPU_NAME${NC}"
fi

BATCH_SIZE=32
GRAD_ACCUM=4
EPOCHS=60
LR=0.001
NUM_WORKERS=6
OUTPUT="models/new/diagnostic_model.pth"
BACKBONE="resnet18"
WEIGHT_DECAY=0.0001

echo ""
echo "🚀 Enhanced Training Configuration:"
echo "  Data:           $FIVEK_DIR"
echo "  Batch size:     $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)) with grad accumulation)"
echo "  Epochs:         $EPOCHS"
echo "  Learning rate:  $LR"
echo "  Backbone:       $BACKBONE"
echo "  Weight decay:   $WEIGHT_DECAY"
echo "  Workers:        $NUM_WORKERS"
echo "  Device:         $DEVICE"
echo "  Output:         $OUTPUT"
echo "  Features:       Mixed Precision, SWA, Focal Loss, Attention"
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

"$PYTHON_BIN" src/training/train_diagnostic_model.py \
    --data "$FIVEK_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    --backbone "$BACKBONE" \
    --gradient-accumulation "$GRAD_ACCUM" \
    --weight-decay "$WEIGHT_DECAY"

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
