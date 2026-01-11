# Assignment 2 — Black-box Adversarial Testing of VGG16 (Hill Climbing)

This repository contains:
- **White-box baselines**: FGM and PGD attacks (CleverHans) in `baselines.py`
- **Black-box attack**: Hill Climbing (HC) in `hill_climbing.py`

Both scripts run on the same input images and labels and write metrics CSV files that are used in the report.

---

## 1) Setup

### Requirements
- Python **3.11**
- Install dependencies:

```bash
pip install -r requirements.txt
```



## 2) Running the Hillclimber

```bash
python hill_climbing.py
```