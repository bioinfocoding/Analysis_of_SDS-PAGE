# SDS-PAGE Analysis

This repository contains code and data for **SDS-PAGE image analysis**.  
It includes functionality to:

- Detect band intensities  
- Perform normalization  
- Calculate **7s and 11s ratios**  
- Apply machine learning classification to predict the class of validation samples  

---

## 📂 File and Folder Overview

- **sds_images/**  
  Contains all SDS-PAGE images (`.jpg` or `.jpeg`).  

- **function.py**  
  Python script for:  
  - Processing SDS-PAGE images  
  - Detecting band intensities  
  - Normalization  
  - 7s/11s ratio identification  
  - Machine learning pipeline  

- **bands_detection.ipynb**  
  Jupyter Notebook to run `function.py`.  

- **classification_file.csv**  
  Contains class labels for each sample.  

- **machine_learning_for_KNN_DT_RF_GB.ipynb**  
  Notebook with ML code for kNN, Decision Tree, Random Forest, and Gradient Boosting.  

- **Generated after running bands_detection.ipynb:**  
  - **image_analysis/** – Each image has its own folder containing:  
    - Detected 4 bands on marker lane  
    - Vertical and horizontal lines  
    - Minimum detected lines per image  
  - **Detected_bands_automatic/** – Automatically detected band images  
  - **manual_7s_11s_picture/** – Manually detected bands across all images  
  - **7s_11s_peak_ranges.csv** – 7s and 11s ratio across all images  
  - **mean_and_sum_of_7s_11s_ratio.csv** – Mean and sum of 7s/11s ratios  
  - **normalized_df.csv** – Normalized intensities of all samples + validation samples  
  - **normalized_samples.csv** – Normalized intensities of samples only  
  - **normalized_validation.csv** – Normalized intensities of validation samples only  
  - **ready_for_ml_class_1.csv** – Final normalized dataset for ML  
  - **ready_for_normalization.csv** – Data before quantile normalization  

---

## 🔄 Workflow Overview

1. **Run `bands_detection.ipynb`**  
   - Generates `image_analysis/` and other processed outputs  

2. **Machine Learning**  
   - Normalized values are used to train models:  
     - SVM  
     - kNN  
     - Random Forest  
     - Decision Tree  
     - Gradient Boosting  
   - Models predict the classes of **validation samples (unlabeled images)**  

3. **7s/11s Ratio**  
   - Ratios are automatically calculated from detected bands  
   - Results saved in:  
     - `7s_11s_peak_ranges.csv`  
     - `mean_and_sum_of_7s_11s_ratio.csv`  

---

## ⚙️ Customization

### Intervals  
By default, the script uses **100 intervals**. To change this, modify line 12 in `function.py`:  

```python
mln = <your_desired_intervals>
```


### Base Directory
By default, outputs are created in the current working directory. To change:

```python
Path.cwd() / "new_directory_name"
```

---

# 📊 Machine Learning Models

We use the following models for classification:

- **Support Vector Machine (SVM)**
- **k-Nearest Neighbors (kNN)**
- **Decision Tree (DT)**
- **Random Forest (RF)**
- **Gradient Boosting (GB)**

Validation set predictions are made using the trained models to classify unlabeled SDS-PAGE images.

---

# 📦 Requirements

- Python 3.x  
- Jupyter Notebook
