# Dementia Stage Classification using MRI and Deep Learning

## Overview

This project develops a machine learning model capable of classifying dementia stages from MRI images. The model aims to assist clinicians by detecting subtle pathological patterns in brain scans that may not be easily observable through traditional analysis.

Three dementia stages are classified:

- **NoImpairment**
- **MildImpairment**
- **ModerateImpairment**

The system is built using **transfer learning with ResNet50V2** and evaluated using multiple performance metrics including accuracy, precision, recall, specificity, F1-score, and ROC curves.

The final model achieved **93.7% test accuracy**, demonstrating strong potential for use as a **clinical decision-support tool**. :contentReference[oaicite:1]{index=1}

---

# Project Objectives

### Technical Objective
Develop a machine learning model capable of classifying dementia stages using MRI images.

### Performance Objective
Achieve performance above chance (>70% accuracy) while evaluating model design and comparing results with existing literature.

### Ethical Objective
Analyse the **benefits, risks, and ethical implications** of using AI in healthcare systems. :contentReference[oaicite:2]{index=2}

---

# Dataset

The dataset contains MRI images categorized into three classes representing dementia severity levels.

| Class | Label | Description |
|------|------|-------------|
| 0 | MildImpairment | Early stage cognitive decline |
| 1 | ModerateImpairment | More advanced cognitive impairment |
| 2 | NoImpairment | No detectable cognitive impairment |

### Dataset Characteristics

- Image format: `uint8`
- Original resolution: **208 × 176 pixels**
- Channels: **3 (RGB)**
- Preprocessed resolution: **224 × 224**

### Class Distribution

| Class | Images | Percentage |
|------|------|-------------|
| NoImpairment | 2560 | 50.4% |
| MildImpairment | 1792 | 35.3% |
| ModerateImpairment | 724 | 14.3% |

The dataset shows **class imbalance**, which was addressed during training using class weights. :contentReference[oaicite:3]{index=3}

---

# Data Preprocessing

The following preprocessing steps were applied:

- Image normalization to `float32`
- Image resizing to **224 × 224**
- Conversion to **3-channel RGB**
- Data split into:

| Dataset | Percentage |
|-------|-----------|
| Training | 80% |
| Validation | 10% |
| Test | 10% |

The validation set was used for hyperparameter tuning and overfitting monitoring. :contentReference[oaicite:4]{index=4}

---

# Model Architecture

The model uses **transfer learning** with a **ResNet50V2 backbone pretrained on ImageNet**.

### Architecture

```
Input (224x224x3)
       ↓
ResNet50V2 (pretrained backbone, classifier removed)
       ↓
GlobalAveragePooling2D
       ↓
Dense (256) + ReLU
       ↓
BatchNormalization
       ↓
Dropout (0.3)
       ↓
Dense (128)
       ↓
Softmax Output (3 classes)
```

Key architectural elements:

- **Transfer learning** for faster convergence and improved generalization
- **Batch Normalization** for training stability
- **Dropout** to reduce overfitting
- **Softmax output** for multi-class classification. :contentReference[oaicite:5]{index=5}

---

# Data Augmentation

Data augmentation was applied **only to the training dataset** to improve generalization.

Augmentation techniques used:

- RandomZoom
- RandomBrightness
- RandomContrast

These transformations help the model learn more robust representations of MRI images. :contentReference[oaicite:6]{index=6}

---

# Model Training

### Training Configuration

- Optimizer: **Adam**
- Loss function: `sparse_categorical_crossentropy`
- Batch size: **32**
- Total epochs: **50**

### Two-Phase Training Strategy

**Phase 1 — Warm-up**

- Backbone frozen
- Learning rate: `1e-4`

**Phase 2 — Fine-tuning**

- Backbone unfrozen
- Learning rate: `1e-5`

### Callbacks

- `EarlyStopping`
- `ReduceLROnPlateau`

These techniques improve convergence and prevent overfitting. :contentReference[oaicite:7]{index=7}

---

# Model Evaluation

The model was evaluated using multiple metrics:

| Metric | Purpose |
|------|---------|
| Accuracy | Overall classification performance |
| Precision | False positive control |
| Recall (Sensitivity) | False negative control |
| Specificity | True negative detection |
| F1-score | Balance between precision and recall |
| ROC / AUC | Class separability |

Confusion matrices and ROC curves were also used to evaluate model behaviour. :contentReference[oaicite:8]{index=8}

---

# Results

### Test Performance

| Class | Precision | Recall | Specificity | F1-score |
|------|----------|-------|-------------|---------|
| MildImpairment | 0.947 | 0.905 | 0.973 | 0.926 |
| ModerateImpairment | 0.895 | 0.932 | 0.982 | 0.913 |
| NoImpairment | 0.943 | 0.961 | 0.940 | 0.952 |

Overall metrics:

- **Accuracy:** 93.7%
- **Macro F1-score:** ~93%
- **Macro Specificity:** ~96%

ROC analysis showed **AUC values close to 1**, indicating strong class separability. :contentReference[oaicite:9]{index=9}

---

# Limitations

Several limitations were identified during model development:

- **Class imbalance**, especially for the ModerateImpairment class.
- Excessive **data augmentation** initially reduced performance.
- Confusion between **MildImpairment and NoImpairment**, which is clinically significant because it may lead to missed early diagnosis.

Future work should include:

- **K-fold cross-validation**
- **Explainability methods (Grad-CAM)** to visualize learned features. :contentReference[oaicite:10]{index=10}

---

# Ethical Considerations

AI in healthcare presents both benefits and risks.

### Benefits

- Detection of subtle pathological patterns
- Reduced clinician workload
- Cost-effective decision support
- Scalability in healthcare systems

### Risks

- Potential misdiagnosis
- Overreliance on AI systems
- Data privacy concerns
- Bias due to non-representative datasets

Therefore, AI should function as a **decision-support tool rather than a replacement for clinicians**. :contentReference[oaicite:11]{index=11}

---

# Future Work

Potential improvements include:

- Implementation of **K-fold cross-validation**
- Integration of **Grad-CAM** for model interpretability
- Testing deeper architectures such as **ResNet152**
- Exploring alternative models such as **VGG19**

These improvements could further enhance performance and clinical reliability. :contentReference[oaicite:12]{index=12}

---

# References

See the full reference list in the original project report for detailed citations on:

- Dementia epidemiology
- MRI-based diagnostics
- Deep learning architectures
- Ethical frameworks for AI in healthcare.
