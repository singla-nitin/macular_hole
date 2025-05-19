# Macular Hole Surgery Outcome Prediction

This repository contains a deep learning pipeline for predicting the **outcome of macular hole surgery** using a combination of **OCT images** and **clinical data**.

## 🧠 About Macular Hole & the Prediction Task

A **macular hole** is a small break in the macula, the central part of the retina responsible for sharp, central vision. Surgical intervention (typically vitrectomy) is used to repair this condition, but the **postoperative visual outcome can vary** significantly between patients.

This model predicts the likelihood of **favorable visual acuity improvement** after surgery based on:

- **Preoperative OCT scans** (retinal cross-sectional images)
- **Clinical data** (e.g., age, preoperative vision, duration of symptoms)

---

## 📁 Repository Structure

```plaintext
macular_hole/
├── main.py             # Training script
├── inference.py        # Inference script for new samples
├── OCTS/               # Folder containing preoperative OCT images
├── clinical_data.csv   # CSV file with associated clinical parameters
├── README.md           # Project overview
└── .gitignore
```
