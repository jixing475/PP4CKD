# 🧬 Drug Repurposing For CKD - Shiny Application

<div align="center">

**AI-Powered Drug Repurposing Platform for Chronic Kidney Disease (CKD)**

[![R Version](https://img.shields.io/badge/R-%E2%89%A5%204.0-blue.svg)](https://www.r-project.org/)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Shiny](https://img.shields.io/badge/Shiny-Web%20App-5a5aff.svg)](https://shiny.rstudio.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [System Requirements](#-system-requirements)
- [Installation Guide](#-installation-guide)
- [User Guide](#-user-guide)
- [Project Structure](#-project-structure)
- [Model Documentation](#-model-documentation)
- [Development Guide](#-development-guide)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🎯 Project Overview

PP4CKD (Protein Prediction Platform for Chronic Kidney Disease) is a deep learning-based drug target prediction platform focused on drug repurposing research for Chronic Kidney Disease (CKD). The platform integrates multiple molecular fingerprint techniques and deep neural network models to provide users with intuitive and efficient drug-target interaction prediction services.

### Background and Significance

- **Chronic Kidney Disease** is a major global health challenge, affecting over 10% of the adult population
- **Drug Repurposing** is an efficient, low-cost drug development strategy
- **AI-Driven** target prediction can significantly accelerate the screening process for candidate compounds

### Core Advantages

- ✅ Multiple molecular fingerprint algorithms (ECFP4/6, MHFP6, RDKit, AtomPair, Layered, Fused)
- ✅ Interactive molecular editor (JSME), no manual SMILES input required
- ✅ Automated SMILES preprocessing and validation
- ✅ Real-time target prediction with confidence scoring
- ✅ Rich statistical analysis and visualization
- ✅ Support for batch processing and result export

---

## ✨ Key Features

### 1. Diverse Molecular Input Methods

| Input Method | Description |
|-------------|-------------|
| **🖌️ Draw Molecule** | Draw chemical structures directly using the JSME molecular editor |
| **⌨️ Manual Input** | Paste or type SMILES strings (batch support) |
| **📁 CSV Upload** | Upload CSV files containing a SMILES column |
| **📚 Example Molecules** | Pre-loaded examples (Caffeine, Ibuprofen, etc.) for quick testing |

### 2. Intelligent SMILES Preprocessing

- Automatic removal of stereochemistry information
- Counter-ion removal
- Valence correction
- Neutralization processing
- Real-time validation feedback

### 3. Multi-Model Target Prediction

Supports 7 types of molecular fingerprint-based deep learning models:

- **ECFP4** - Morgan fingerprint (radius=2, 4096 bits)
- **ECFP6** - Morgan fingerprint (radius=3, 4096 bits)
- **AtomPair** - Atom pair fingerprint (4096 bits)
- **Layered** - RDKit layered fingerprint (4096 bits)
- **RDKit** - RDKit topological fingerprint (4096 bits)
- **MHFP6** - MinHash fingerprint (radius=3, 4096 bits)
- **Fused** - ECFP4+MHFP6 fused fingerprint (8192 bits)

### 4. Results Analysis and Visualization

- 📊 Confidence distribution by rank
- 🧬 Protein class distribution
- 🦠 Organism distribution (pie chart)
- 🎯 Target type distribution (pie chart)
- 📋 Interactive results table (sortable, searchable)
- 🖼️ Molecular structure hover preview

### 5. Data Export

- **CSV Format** - Complete prediction results
- **Excel Format** - MS Office compatible
- **Top-5 Summary** - Top 5 predicted targets for each molecule

---

## 🛠️ Technology Stack

### Backend Technologies

- **R Shiny** - Web application framework
- **reticulate** - R-Python interface
- **RDKit** - Cheminformatics toolkit (Python)
- **scikit-learn** - Machine learning models (joblib serialization)

### Frontend Technologies

- **JSME** - JavaScript molecular editor
- **DT (DataTables)** - Interactive tables
- **Plotly** - Interactive visualization
- **Tippy.js** - Tooltips
- **Custom CSS** - Responsive design

### Data Science

- **NumPy** - Numerical computation
- **Pandas** - Data processing
- **MHFP** - MinHash fingerprint generation

---

## 💻 System Requirements

### Minimum Configuration

- **Operating System**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **R Version**: ≥ 4.0
- **Python Version**: 3.8 - 3.10
- **Memory**: 8 GB RAM (16 GB recommended)
- **Disk Space**: 5 GB (including model files)

### Recommended Configuration

- **Memory**: 16 GB+ RAM
- **Processor**: Multi-core CPU (faster predictions)
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+

---

## 📦 Installation Guide

### Method 1: Using Python Virtual Environment (Recommended)

#### Step 1: Clone or Download the Project

```bash
cd ~/Desktop
unzip PP4CKD.zip  # or use git clone
cd PP4CKD
```

#### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv-shiny

# Activate virtual environment
# macOS/Linux:
source .venv-shiny/bin/activate
# Windows:
.venv-shiny\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install numpy pandas rdkit mhfp scikit-learn joblib
```

#### Step 3: Install R Packages

Open R or RStudio and run:

```r
# Install required R packages
install.packages(c(
  "shiny",
  "reticulate", 
  "tidyverse",
  "DT",
  "plotly",
  "shinyjs",
  "writexl"
))

# Or run the project's check script
source("scripts/check_packages.R")
```

#### Step 4: Verify Installation

```bash
# Check Python environment
.venv-shiny/bin/python --version
.venv-shiny/bin/python -c "import rdkit; print('RDKit OK')"

# Check R packages
Rscript scripts/check_packages.R
```

### Method 2: Using Conda Environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate pp4ckd

# Install R packages (execute in R)
Rscript scripts/check_packages.R
```

---

## 🚀 User Guide

### Launching the Application

#### Method 1: Using Launch Script (Recommended)

```bash
Rscript launch_app.R
```

The application will automatically open in your browser (default port: 8080)

#### Method 2: Launch from R

```r
library(shiny)
runApp(".", port = 8080, launch.browser = TRUE)
```

### Workflow

#### 1️⃣ Input Molecules

**Option A - Draw Molecule:**
1. Click the "Draw Molecule" tab
2. Use the JSME editor to draw molecular structures
3. Click "Validate SMILES" to validate

**Option B - Manual Input:**
1. Click the "Manual Input" tab
2. Enter one SMILES string per line
3. Click "Validate SMILES" to validate

**Option C - Upload CSV:**
1. Click the "Upload CSV" tab
2. Upload a CSV file containing a `SMILES` column (refer to `data/example_input.csv` format)
3. Click "Validate SMILES" to validate

**Option D - Use Examples:**
1. Click the "Example Molecules" tab
2. Select a preset example or click "Load All from example_input.csv"

#### 2️⃣ Validate SMILES

- Click the "Validate SMILES" button
- View the validation results popup:
  - ✅ Green: Valid SMILES
  - ❌ Red: Invalid SMILES
- The system automatically preprocesses valid SMILES (removes stereochemistry, neutralizes, etc.)

#### 3️⃣ Configure Model

- Select fingerprint type (ECFP4, ECFP6, MHFP6, Fused, etc.)
- Set number of predictions (Top-N, range: 1-20)

#### 4️⃣ Execute Prediction

- Click the "Predict Targets" button
- Wait for calculation to complete (progress bar indicator)
- Automatically navigate to results page

#### 5️⃣ Analyze Results

**Summary Statistics:**
- Total number of molecules
- Number of unique predicted targets
- Average confidence score
- Model type used

**Visualization Charts:**
- Confidence distribution by rank
- Protein class distribution
- Organism distribution
- Target type distribution

**Detailed Prediction Table:**
- Molecule index and SMILES
- Target rank
- ChEMBL target ID (clickable for details)
- Target name
- Confidence score (0-1)
- Protein classification, type, organism
- Hover over SMILES to display molecular structure

#### 6️⃣ Export Results

- **Download CSV** - Complete prediction results
- **Download Excel** - Excel format
- **Download Top-5 Summary** - Top 5 targets for each molecule

---

## 📂 Project Structure

```
PP4CKD/
│
├── 📄 app.R                        # Main Shiny application
├── 📄 launch_app.R                 # Application launch script
├── 📄 README.md                    # Project documentation (this file)
│
├── 📁 scripts/                     # Script folder
│   ├── 03-fingerprints.py         # Molecular fingerprint calculation functions
│   ├── 05-preprocessing.py        # SMILES preprocessing and validation
│   ├── 06-utils.py                # Utility functions
│   └── check_packages.R           # R package dependency checker
│
├── 📁 data/                        # Data folder
│   ├── example_input.csv          # Example input data
│   └── external/                  # External reference data
│       ├── DNNTARLABELS_2nd.txt         # Target label list
│       ├── TARGETCLASSIFICATION_2nd.txt # Target classification info
│       └── TARGETSDETAILS_2nd.txt       # Target details
│
├── 📁 model/                       # Model folder
│   ├── ecfp4_dnn_model_full_data.joblib   # ECFP4 model
│   ├── fused_dnn_model_full_data.joblib   # Fused model
│   ├── (other model files...)
│   └── README_MODELS.md                   # Model documentation
│
├── 📁 www/                         # Web static assets
│   ├── JSME_2024-04-29/           # JSME molecular editor library
│   ├── jsme_local.html            # JSME embedded page
│   ├── molecules/                 # Generated molecule image cache
│   ├── styles.css                 # Application stylesheet
│   └── tippy-config.js            # Tooltip configuration
│
├── 📁 .venv-shiny/                 # Python virtual environment
│   ├── bin/                       # Python executables
│   ├── lib/                       # Python libraries
│   └── pyvenv.cfg                 # Virtual environment config
│
└── 📁 Configuration Files
    ├── environment.yml            # Conda environment config
    ├── requirements.txt           # Python dependencies list
    └── AGENTS.md                  # AI agent config (for development)
```

---

## 🤖 Model Documentation

### Model Architecture

- **Type**: Multi-Output Deep Neural Network (Multi-Output DNN)
- **Framework**: scikit-learn MultiOutputClassifier
- **Input**: Molecular fingerprint vectors (4096 or 8192 dimensions)
- **Output**: Multi-label binary classification probabilities (one probability per target)

### Supported Fingerprint Types

| Fingerprint Type | Dimensions | Features | Use Case |
|-----------------|-----------|----------|----------|
| **ECFP4** | 4096 | Morgan fingerprint, radius=2 | General purpose, balanced speed and performance |
| **ECFP6** | 4096 | Morgan fingerprint, radius=3 | Larger molecular neighborhood information |
| **AtomPair** | 4096 | Atom pair topological distance | Captures inter-atomic distance information |
| **Layered** | 4096 | RDKit layered fingerprint | Considers hierarchical structure of molecular graph |
| **RDKit** | 4096 | RDKit topological fingerprint | Based on substructure paths |
| **MHFP6** | 4096 | MinHash fingerprint | Efficient similarity search |
| **Fused** | 8192 | ECFP4+MHFP6 fusion | Best performance but higher computational cost |

### Model Performance

- **Training Data**: Compound-target interaction data from ChEMBL database
- **Number of Targets**: 9 CKD-related targets
- **Evaluation Metrics**: AUC-ROC, Precision, Recall, F1-Score

### Adding New Models

1. Train the model and save in `.joblib` format:
   ```python
   import joblib
   joblib.dump(model, 'model/your_model_dnn_model_full_data.joblib')
   ```

2. Ensure model naming follows the convention: `{fingerprint_type}_dnn_model_full_data.joblib`

3. Add the new fingerprint type to the `model_types` vector in `app.R`

4. Add the corresponding fingerprint calculation function to `FINGERPRINT_FUNCTIONS` in `scripts/03-fingerprints.py`

---

## 🔧 Development Guide

### Development Environment Setup

```bash
# Clone project
cd ~/projects
git clone <repository_url> PP4CKD
cd PP4CKD

# Setup Python environment
python3 -m venv .venv-shiny
source .venv-shiny/bin/activate
pip install -r requirements.txt

# Setup R environment (execute in R)
install.packages(c("shiny", "reticulate", "tidyverse", "DT", "plotly", "shinyjs", "writexl"))
```

### Code Structure

#### R Side (app.R)

```r
# 1. Environment configuration
Sys.setenv(RETICULATE_PYTHON = ".venv-shiny/bin/python")

# 2. Load libraries
library(shiny); library(reticulate); ...

# 3. Load Python modules
source_python("scripts/03-fingerprints.py")
source_python("scripts/05-preprocessing.py")
source_python("scripts/06-utils.py")

# 4. Load data and models
target_details <- read.delim("data/external/TARGETSDETAILS_2nd.txt")
MODELS <- list(...)  # Load joblib models

# 5. UI definition
ui <- fluidPage(...)

# 6. Server logic
server <- function(input, output, session) {
  # Reactive values
  rv <- reactiveValues(...)
  
  # Event handlers
  observeEvent(input$validate_btn, {...})
  observeEvent(input$predict_btn, {...})
  
  # Output rendering
  output$results_table <- renderDT({...})
}

# 7. Launch app
shinyApp(ui, server)
```

#### Python Side (scripts/*.py)

- **03-fingerprints.py**: Implements various molecular fingerprint calculation functions
- **05-preprocessing.py**: SMILES preprocessing and validation
- **06-utils.py**: Utility functions

### Adding New Features

#### Example: Adding a New Fingerprint Type

1. **Add calculation function in `scripts/03-fingerprints.py`**:
   ```python
   def calculate_new_fingerprint(smiles, n_bits=4096):
       mol = Chem.MolFromSmiles(smiles)
       fp = YourFingerprintFunction(mol, nBits=n_bits)
       return fp
   
   FINGERPRINT_FUNCTIONS['NewFP'] = calculate_new_fingerprint
   ```

2. **Add model type in `app.R`**:
   ```r
   model_types <- c('ECFP4', 'ECFP6', ..., 'NewFP')
   ```

3. **Train and save the model**:
   ```python
   joblib.dump(model, 'model/newfp_dnn_model_full_data.joblib')
   ```

### Debugging Tips

- **View Python output**: Check Python `print()` output in the R console
- **Use `browser()`**: Insert breakpoints in R code
- **Check reactive values**: Use `observe({ print(rv$variable) })`
- **Chrome DevTools**: Inspect frontend JavaScript errors

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Application Won't Start

**Error**: `Could not find python at '.venv-shiny/bin/python'`

**Solution**:
```bash
# Check if Python environment exists
ls .venv-shiny/bin/python

# If not, recreate it
python3 -m venv .venv-shiny
source .venv-shiny/bin/activate
pip install -r requirements.txt
```

#### 2. Missing R Packages

**Error**: `Error in library(shiny) : there is no package called 'shiny'`

**Solution**:
```r
# Run package check script
source("scripts/check_packages.R")

# Or manually install missing packages
install.packages("missing_package_name")
```

#### 3. RDKit Import Error

**Error**: `No module named 'rdkit'`

**Solution**:
```bash
# Activate virtual environment
source .venv-shiny/bin/activate

# Install RDKit
pip install rdkit

# If pip installation fails, use conda
conda install -c conda-forge rdkit
```

#### 4. Model Loading Failed

**Error**: `Error loading ECFP4: No such file or directory`

**Solution**:
- Check if model files exist in the `model/` directory
- Confirm file naming follows the format: `{type}_dnn_model_full_data.joblib`
- Review `model/README_MODELS.md` for model requirements

#### 5. JSME Molecular Editor Not Displaying

**Solution**:
- Check if the `www/JSME_2024-04-29/` directory is complete
- Check browser console for JavaScript errors
- Try clearing browser cache and refreshing

#### 6. Slow Prediction Speed

**Optimization Suggestions**:
- Use smaller fingerprint types (avoid Fused)
- Reduce number of predictions (set Top-N to 5 instead of 20)
- Upgrade hardware (more RAM and CPU cores)
- Use GPU acceleration (requires modifying models to TensorFlow/PyTorch)

### Logging and Debugging

#### Enable Verbose Logging

Add to the top of `app.R`:
```r
options(shiny.trace = TRUE)
options(shiny.error = browser)
```

#### View Python Output

```r
py_run_string("import sys; print('Python version:', sys.version)")
py_run_string("import rdkit; print('RDKit version:', rdkit.__version__)")
```

---

## 🤝 Contributing

We welcome community contributions! Please follow these steps:

### Reporting Issues

1. Search the [Issues page](../../issues) for similar existing issues
2. Create a new Issue with detailed information:
   - Operating system and version
   - R and Python versions
   - Error messages and stack traces
   - Steps to reproduce

### Submitting Code

1. Fork this project
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

### Code Style

- **R code**: Follow the [Tidyverse Style Guide](https://style.tidyverse.org/)
- **Python code**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Commit messages**: Use clear, descriptive commit messages

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📚 Citation

If you use this platform in your research, please cite:

```bibtex
@software{pp4ckd2024,
  title = {PP4CKD: Drug Repurposing Platform for Chronic Kidney Disease},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/PP4CKD}
}
```

---

## 📧 Contact

- **Project Maintainer**: [Your Name]
- **Email**: your.email@example.com
- **Issue Reporting**: [GitHub Issues](../../issues)

---

## 🙏 Acknowledgments

- **RDKit** - Open-source cheminformatics toolkit
- **ChEMBL** - Bioactivity database
- **JSME** - JavaScript molecular editor
- **Shiny** - R web application framework
- **All Contributors** - Thanks to the community for support

---

<div align="center">

**⭐ If this project helps you, please give us a Star!**

Made with ❤️ for Drug Discovery

</div>
