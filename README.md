# SDS-PAGE Analysis

This repository contains code and data for **SDS-PAGE image analysis** to:
- Detect and measure the **intensity of bands**
- Calculate **7s and 11s subunit ratios**
- Perform **machine learning classification** to predict the class of validation samples

---

## 📁 Files Overview

| File/Folder | Description |
|-------------|-------------|
| `function.py` | Python script for processing SDS-PAGE images. Includes: band intensity calculation, normalization, 7s/11s ratio identification, and classification using machine learning (SVM). |
| `bands_detection.ipynb` | Jupyter Notebook interface to run functions from `function.py`. |
| `classification_file.csv` | Contains class labels of each sample. |
| `*.jpg / *.jpeg` | Input SDS-PAGE image files. |

---

## 🔁 Workflow Overview

1. **Run `bands_detection.ipynb`** to begin analysis.
2. The notebook will:
   - Process each input image
   - Create a folder per image containing:
     - Detected marker bands (4 bands)
     - Vertical and horizontal alignment lines
     - Bands detected from each lane
3. Outputs will be organized into the following folders:

| Folder | Description |
|--------|-------------|
| `7s_11s_picture/` | Automatically detected 7s and 11s subunit bands across all images. |
| `manual_7s_11s_picture/` | Manual detection results for 7s/11s ratios. |
| `mean/` | CSV files containing the mean intensity values for detected bands. |
| `sum/` | CSV files containing the sum of intensity values for each band. |
| `Peak_output/` | Images of peak detection results and corresponding peak range details. |

---

## 🤖 Notes

- **Machine Learning Model: **: We use an SVM for classification based on the intensity values extracted from the SDS-PAGE images.
- **Validation**: After training, the model is used to predict the classes of the 2 images without labels in the validation set.

---

## 📦 Requirements

- Python 3.x
- Jupyter Notebook

### Required Python Libraries:
```python
os
cv2
pandas
matplotlib
ast
numpy
scikit-learn
joblib
