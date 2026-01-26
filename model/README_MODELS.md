# Model Files Documentation

## 📁 Model File Location

Please place trained scikit-learn model files in this directory.

## 📝 File Naming Convention

Model files should follow this naming format:

```
{fingerprint_type}_dnn_model_full_data.{format}
```

### Supported Formats

- **`.pkl`** - Python pickle format (**Recommended**, smaller file size)
- **`.joblib`** - joblib format (optimized for large numpy arrays)

### Example File Names

```
model/
├── ecfp4_dnn_model_full_data.pkl    # ECFP4 model
├── ecfp6_dnn_model_full_data.pkl    # ECFP6 model
├── atompair_dnn_model_full_data.pkl # AtomPair model
├── layered_dnn_model_full_data.pkl  # Layered model
├── rdkit_dnn_model_full_data.pkl    # RDKit model
├── mhfp6_dnn_model_full_data.pkl    # MHFP6 model
└── fused_dnn_model_full_data.pkl    # Fused model
```

## ⚙️ Configuration

Set in the YAML parameters of QMD file:

```yaml
params:
  model_type: "ECFP4"       # Select model to use
  model_dir: "model"        # Model directory
  model_format: "pkl"       # Model format: pkl or joblib
```

## 🔍 Model Requirements

- **Model Type**: scikit-learn `MultiOutputClassifier`
- **Input**: Molecular fingerprint vector (dimension depends on fingerprint type)
- **Output**: Multi-label classification probabilities (corresponding to target labels)

## 📊 Fingerprint Types and Dimensions

| Fingerprint Type | Dimensions | Description |
|-----------------|-----------|-------------|
| ECFP4 | 4096 | Morgan fingerprint (radius=2) |
| ECFP6 | 4096 | Morgan fingerprint (radius=3) |
| AtomPair | 4096 | Atom pair fingerprint |
| Layered | 4096 | RDKit layered fingerprint |
| RDKit | 4096 | RDKit topological fingerprint |
| MHFP6 | 4096 | MinHash fingerprint (radius=3) |
| Fused | 8192 | ECFP4 + MHFP6 fusion |

## ⚠️ Notes

1. **Model format consistency**: Ensure all model files use the same format (pkl or joblib)
2. **Target label order**: Model output order must be consistent with `external/DNNTARLABELS_2nd.txt`
3. **File size**: Fused model files may be large (due to 8192-dimensional input)

## 🔧 Generate Placeholder Model Files (for testing)

If you need to generate fake model files for testing QMD rendering, you can run:

```python
# Example: Generate fake ECFP4 model
import pickle
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

# Create a simple placeholder model
n_features = 4096  # ECFP4 dimensions
n_targets = 10     # Number of targets

base_model = LogisticRegression(max_iter=10)
model = MultiOutputClassifier(base_model)

# Train with fake data (just for initialization)
X_fake = np.random.rand(50, n_features)
y_fake = np.random.randint(0, 2, size=(50, n_targets))
model.fit(X_fake, y_fake)

# Save model
with open('model/ecfp4_dnn_model_full_data.pkl', 'wb') as f:
    pickle.dump(model, f)
```

## ✅ Verify Model Files

```bash
# Check if model files exist
ls -lh model/*_dnn_model_full_data.*

# Test loading in Python
python3 -c "import pickle; model = pickle.load(open('model/ecfp4_dnn_model_full_data.pkl', 'rb')); print('Model loaded:', model)"
```
