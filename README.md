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
- Output: Opens a preview window that shows original vs. corrected along with scene + parameter details and a clickable filter
  gallery (fused result + any recommended presets). The selected filter is what gets displayed. A copy of the fused correction
  is written to `data/output/scene_corrected.jpg`, and recommended filter previews (Natural/Vivid/Muted) are saved under
  `data/output/presets/`.
  The script also verifies that the same parameters drive both the preview and manual correction pipeline and reports the
  observed pixel-level change, ensuring the saved image reflects the intended adjustments.

**How `--fusion_weight` affects the result**

- Higher values (closer to **1.0**) lean more on the AI diagnostic model, so exposure/white balance/saturation shifts follow
  what the diagnostic network recommends for the specific photo. Use this when the scene preset looks off (e.g., misclassified
  scene) or you want more adaptive corrections.
- Lower values (closer to **0.0**) trust the scene preset more, so the output follows the typical look for that scene category
  and reduces the influence of the diagnostic network. Use this when the preset already matches the scene style and you want a
  more conservative change.
- The default **0.6** is a balanced mix: it keeps the preset’s intended look while still letting the diagnostic model nudge the
  parameters toward the measured exposure/white balance/saturation needs of the input image.

---
# 🛠️ Visual Studio Submission (For Graders)

To comply with the assignment submission requirement ("Submit source codes... in an integrated environment (Visual Studio)"):
We primarily developed in VS Code for Python compatibility.
For submission, we have generated a Visual Studio Solution (.sln).
You can open the .sln file in Visual Studio 2022 (with Python workload installed) to run the project.

It automatically detects the conda environment `DIP_Project2` that has been created before and uses it for the dependencies.
A wrapper python file `main.py` is set up as startup file, as it is used to execute the same commands as the console helper would do.

---

## 🐍 Fixing Visual Studio Conda Environment Issues

If Visual Studio shows an error like:

> Failed to launch debugger.  
> The environment `CondaEnv|CondaEnv|DIP_Project2` appears to be incorrectly configured or missing.

follow these steps to re-connect the existing Conda environment `DIP_Project2` to Visual Studio.

### 1. Remove the broken Python environment entry in Visual Studio

1. In the top menu, go to **View → Other Windows → Python Environments**.
2. In the list, look for an entry such as **`CondaEnv|CondaEnv|DIP_Project2`**.
3. Select it and click **Remove**.  
   > This only deletes the registration inside Visual Studio – the actual Conda environment on disk remains intact.

### 2. Re-add the existing Conda environment `DIP_Project2`

1. In the **Python Environments** window, click the **`+` (Add Environment)** button.
2. Choose **Existing environment**.
3. Set **Environment type** to **Conda**.
4. For the **Interpreter** path, browse to your environment’s Python executable, for example:
   ```text
   C:\Users\jimmy\miniconda3\envs\DIP_Project2\python.exe
   ```
5. Give it a clear name such as DIP_Project2 and click Create/Add.

### 3. Set DIP_Project2 as the project interpreter and run

1. In Solution Explorer, right-click on the project and select **Properties**.
2. Under the **General** tab, set the **Python Environment** to the newly added **DIP_Project2**.
3. Save and close the properties window.
4. Now press F5 or click the Run button – the project should start using the Conda environment DIP_Project2 without the previous debugger error.