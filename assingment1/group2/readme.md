# Group 2 – Citizen Welfare Fraud Risk Models

This repository contains the code and artefacts for Group 2:

- **pipeline_models_group2/**  
  Training code for the *good* and *bad* models, plus our own automated tests.
- **cross_test_models/**  
  Other groups' testing pipelines applied to our models (cross-group evaluation).
- **model_1.onnx** and **model_2.onnx**  
  Exported ONNX versions of our good and bad models, respectively.

The main model is a `GradientBoostingClassifier` trained on the provided Rotterdam dataset, with feature engineering and bias-analysis helpers in `group2_helpers.py`.

---

## 1. How to set up the environment

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate
   
2. Install dependencies:
   ``` pip install -r requirements.txt```
3. To train and run the automated tests run the notebook : pipeline_models_group2.ipynb
4. To run the tests of the other team on our model run: cross_test_models.ipynb