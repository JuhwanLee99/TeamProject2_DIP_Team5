# 🎨 Image Color Harmony Adjuster  
**Team 05**

This project was developed for the **Digital Image Processing** course.  
The goal is to build a system that **automatically analyzes and corrects color imbalances** in digital images caused by various lighting conditions.

---

## Project Logic
1. Core DIP Logic (Manual & Differentiable):
The src/correction.py file contains manually implemented DIP functions (e.g., apply_gamma, apply_white_balance). These functions are written using PyTorch tensor operations instead of NumPy, making them fully differentiable. This satisfies the "own implementation" requirement.

2. ML as a Supplementary Tool (Automatic):
The src/model.py defines a Deep Learning model (CNN) that acts as a "supplementary tool." It analyzes an image and predicts the optimal parameters (e.g., gamma=1.15, saturation=1.1) required by the manual functions.

3. Process:
- train.py: Teaches the model to predict parameters. The model's predicted parameters are fed into the src/correction.py functions to generate a corrected image inside the training loop. The loss is calculated between this corrected image and the ground-truth image, allowing the model to be trained via backpropagation.

- predict.py: Uses the trained model to get parameters for a new image, then feeds those parameters into the same manual src/correction.py functions to get the final, corrected result.

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
/DIP_Project2/         
│
├── .gitignore                
├── README.md                
├── environment.yml           # library dependencies
│
├── train.py                  # <<< Script to train the model
├── predict.py                # <<< Script to run prediction on new images
│
├── src/                      # <<< ALL SOURCE CODE MODULES
│   ├── dataset.py            # Loads and transforms training data (image pairs)
│   ├── model.py              # ML model architecture (predicts parameters)
│   ├── correction.py         # <<< CORE: Manual DIP logic (implemented in PyTorch)
│   ├── io_utils.py           # OpenCV functions for reading/saving images
|   ├── conversions.py        # 
│   └── visualization.py      # Matplotlib functions for plotting results
│
├── data/
│   ├── INTEL-TAU/            # (Example) Dataset root
│   │   ├── ... (image files)
│   │
│   ├── input/                # Add sample images for testing here
│   │   └── sample1.jpg
│   └── output/               # Corrected images (ignored by Git)
|
├── models/                   # Folder for saved model checkpoints (ignored by Git)
|    └── .gitignore           # Ignores large .pth (model weight) files
|
└── docs/                     # Project documents (reports, plans)
    └── Survey_and_Plan_Report.docx
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

## Download Dataset
Download the dataset (e.g., INTEL-TAU, Cube++) and place it in the data/ directory. You will need to update src/dataset.py to point to the correct file paths.

## ▶️ How to Run the Program
Ensure your dip-hybrid Conda environment is active.
### Phase 1: Train the Model
Run train.py. The model will learn to output a vector of parameters.

```bash
python train.py
```
This will save a hybrid_model.pth file in the models/ directory.

### Phase 2: Predict with the Hybrid Model
Run predict.py. This script will:
1. Load the trained model (models/hybrid_model.pth).
2. Model predicts parameters (e.g., [gamma: 1.15, r_gain: 1.05, ...]).
3. Script passes these parameters to the manual src/correction.py functions.
4. Show/save the final corrected image.

```bash
python predict.py --model_path models/hybrid_model.pth --input_path data/input/sample1.jpg
```
---