# 🎨 AI-Powered Image Color Harmony Adjuster
**Team 05 (Global Team)**

> **Digital Image Processing Team Project #2** > A hybrid system that automatically analyzes and corrects color imbalances using **Deep Learning** for parameter estimation and **Manually Implemented Algorithms** for image correction.

---

### 1. ✅ Core Logic (Manual Implementation)
We **manually implemented** the fundamental image processing algorithms using **PyTorch tensor operations**. These functions are fully differentiable.
* **Location:** `src/correction.py`, `src/conversions.py`
* **Implemented Algorithms:**
    * **Color Space Conversion:** RGB $\leftrightarrow$ HSV (Manual formula implementation)
    * **White Balance:** Gray-World Assumption based Gain adjustment
    * **Gamma Correction:** Power-law transformation ($I_{out} = I_{in}^{\gamma}$)
    * **Saturation/Hue Adjustment:** Pixel-wise tensor manipulation

### 2. 🧠 AI as a Supplementary Tool (Automatic Analysis)
We reuse **MobileNetV2** to classify the scene, then fuse its preset with an AI diagnostic optimizer that estimates exposure, white balance, and saturation tweaks.
* **Location:** `src/ai_scene.py`, `src/ai_optimize.py`
* **Mechanism:**
    * **Scene Classification:** MobileNetV2 predicts a coarse scene label (e.g., Landscape, Portrait) and preset correction parameters.
    * **AI Diagnostics:** `ai_optimize.optimize_corrections` loads a lightweight CNN (`models/diagnostic_model.pth`) that analyzes exposure, white balance, and saturation, then blends the result with the scene preset.
* **Benefit:** Lightweight AI determines _how much_ to adjust while the manual pipeline still applies every pixel-level edit.

---

## 🌿 Branch naming

- feature/<desciption>: Developing a new feature
- bugfix/<desciption>: Fixing a bug
- hotfix/<desciption>: Applying an urgent fix

```bash
Examples
feature/ocr-text-highlighting
bugfix/pdf-render-error 
```

---

## 📝 Commit Message Format

```bash
<type>: <description>
[optional: body]
```

- type: feat, fix, docs, style, refactor, test, chore

- 예시
```text
feat: Implement null algorithm

- Show detailed image with modified result
- Add related test cases  
```

---

## ✅ Review & Merge

- Each PR requires approval from at least two reviewers
- Merge into the develop branch after addressing review comments
- Merging into the main branch is handled by the release manager

---

## 📂 Project Structure

The project follows a modular design, organizing all logic into separate modules under `src/`.

```markdown
DIP_Team05_Project/
│
├── .gitignore
├── README.md
├── environment.yml           # Conda environment dependencies
├── requirements.txt          # Pip dependencies (alternative)
│
├── app.py                    # for streamlit web demo
├── predict.py                # for console image edit script
│
├── src/                      # <<< SOURCE CODE MODULES
│   ├── ai_scene.py           # CORE: Scene classification & static presets provided (MobileNetV2)
│   ├── ai_optimize.py        #  AI diagnostics + fusion of presets into final correction parameters
│   ├── correction.py         # <<< CORE: Manual DIP algorithms (Differentiable)
│   ├── conversions.py        # <<< CORE: Manual RGB<->HSV conversion logic
│   ├── analysis.py           # for analyze histogram
│   ├── io_utils.py           # OpenCV I/O wrapper
│   └── visualization.py      # Matplotlib visualization tools
│
├── data/
│   ├── input/                # Test images
│   └── output/               # Result images
```

--- 

# 🚀 Getting Started

Follow these instructions to set up your local development environment. This workflow ensures a 100% identical environment for all team members (Mac, Linux, Windows).

## 1. Prerequisites

Ensure you have Miniconda or Anaconda installed on your system.

## 2. Clone the Repository

Clone the GitHub repository and navigate into the project directory.

```bash
git clone <your-github-repo-url>
cd DIP_Project2
```

## 3. Create and Activate the Conda Environment

You will create the entire environment from the environment.yml file. This installs Python, PyTorch, OpenCV, and all other dependencies in one step.

```bash
# 1. Create the environment from the file
# This may take a few minutes
conda env create -f environment.yml

# 2. Activate the newly created environment
conda activate DIP_Project2
```
(Your terminal prompt should now show (DIP_Project2).)

## 4. Configure VS Code (Recommended)
1. Open the project folder in VS Code (code .).
2. Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P).
3. Type Python: Select Interpreter.
4. VS Code should automatically detect and suggest the DIP_Project2 Conda environment. Select it. (It will look something like .../miniconda3/envs/DIP_Project2/bin/python).

---

# ▶️ Usage Guide

The console helper runs scene classification with MobileNetV2 and then applies the preset corrections to the input image.

### 5. Download MobileNetV2 weights (once per machine)

The scene classifier loads a plain PyTorch checkpoint (no training step required). Run the following helper to download the
ImageNet-pretrained weights and store them under `weights/`:

```bash
# Create a directory to keep model weights
mkdir -p weights

# Download the ImageNet-pretrained MobileNetV2 checkpoint (~14 MB)
python - <<'PY'
from pathlib import Path

import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

weights_path = Path("weights/mobilenet_v2-b0353104.pth")
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
torch.save(model.state_dict(), weights_path)
print(f"Saved weights to {weights_path.resolve()}")
PY
```

### 6. Run the console helper (scene + diagnostic AI)

```bash
python predict.py \
  --weights_path weights/mobilenet_v2-b0353104.pth \
  --input_path data/input/your_image.jpg \
  --fusion_weight 0.6
```

- `--weights_path`: MobileNetV2 ImageNet checkpoint for scene classification.
- `--input_path`: Path to your image (RGB/BGR supported).
- `--fusion_weight`: Blend between AI diagnostics and preset (1.0 = AI-only, 0.0 = preset-only). Defaults to 0.6.
- `--diagnostic_model_path` (optional): Custom path to the diagnostic CNN. Defaults to the bundled `models/diagnostic_model.pth`.
- Output: Opens a side-by-side preview and writes `data/output/scene_corrected.jpg`. Recommended filter previews (Natural/Vivid/Muted) are saved under `data/output/presets/`.

---
# 🛠️ Visual Studio Submission (For Graders)

To comply with the assignment submission requirement ("Submit source codes... in an integrated environment (Visual Studio)"):
We primarily developed in VS Code for Python compatibility.
For submission, we have generated a Visual Studio Solution (.sln).
You can open the .sln file in Visual Studio 2022 (with Python workload installed) to run the project.
