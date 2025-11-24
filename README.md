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
Instead of training a custom CNN, we reuse **MobileNetV2** to classify the scene and pick a preset of correction parameters.
* **Location:** `src/ai_scene.py, ai_correction_advisor.py`
* **Mechanism:** The model analyzes the input image to infer a coarse scene label (e.g., Landscape, Portrait) and returns pre-defined parameters that feed into the manual correction pipeline.
* **Benefit:** This keeps the AI component lightweight while ensuring all pixel-level edits still come from the manual code.

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
│   ├── ai_correction_advisor.py #  Calculate diagnostic and final calibration values (Advisor)
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

```Bash
python predict.py --weights_path /path/to/mobilenet_v2-b0353104.pth --input_path data/input/sample1.jpg
```
- `--weights_path` should point to the locally downloaded MobileNetV2 weights file.
- Output: Displays "Original vs Corrected" comparison and saves the result to `data/output/`.

---
# 🛠️ Visual Studio Submission (For Graders)

To comply with the assignment submission requirement ("Submit source codes... in an integrated environment (Visual Studio)"):
We primarily developed in VS Code for Python compatibility.
For submission, we have generated a Visual Studio Solution (.sln).
You can open the .sln file in Visual Studio 2022 (with Python workload installed) to run the project.
