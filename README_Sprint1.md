
# README — Sprint 1: System Design  
### Project: Predicting Product Return Risk in Retail Using Classification

## 1. Overview
In retail and e-commerce, product returns contribute significantly to operational and financial losses. This project aims to design a machine learning system that predicts whether an order will be returned or cancelled at the time of purchase.

## 2. Business Problem
Retailers lose money on logistics, refunds, and product handling due to returns. Predicting return risk at order time allows interventions such as confirmation prompts, fraud detection, recommendation adjustments, and optimized shipping decisions.

**Business Question:** “Will this item/order be returned?”

## 3. Machine Learning Problem
- **Type:** Supervised Learning — Binary Classification  
- **Target Variable:**  
  - `1` → Returned or Cancelled  
  - `0` → Not Returned  
- **Input Features:** age, gender, state, quantity, price, discount, product rating, brand

The objective is to train a classifier that performs significantly better than simple baselines.

## 4. Dataset Scouting & Selection
Two potential datasets were considered:

### Candidate Dataset 1: Product Return Risk Prediction
- Contains age, gender, state, quantity, price, discount, product rating, brand  
- Clean return labels  
- Good volume and structure

### Candidate Dataset 2: Online Retail Dataset (UCI)
- This dataset had less numbers of features.
- It does not had labellings.


### Selected Dataset: Product Return Risk Prediction 
**Reason:** Clear return flags, strong feature coverage, discount and suitability for classification.

## 5. Label Definition
```
1 = Yes
0 = No
```

## 6. Key Features
- **Product/Order:** Category, price, discount, quantity, Product Rating, Brand
- **Customer:** Customer ID, Gender, State, region , Age

These are known correlates of return behavior.

## 7. High-Level System Architecture
- **Python** for data handling and ML (pandas, sklearn, XGBoost, Random Forest, Logistic Regression)  
- **StreamLit** for model serving  
- **joblib** for model serialization
- 

Flow:
1. UI collects order details  
2. Model Loads & Predicts  
3. UI displays return Yes/No
 
## 8. System Architecture Diagram
```
Front-End UI
     │
     ▼
 StreamLit
     │
     ▼
ML Model (.pkl)
     │
     ▼
Prediction Output (Return / Not Return)
```
<img width="1019" height="556" alt="image" src="https://github.com/user-attachments/assets/d641e5bd-e213-49c1-9051-c63c02186f6d" />

## 9. Completed Deliverables (Sprint 1)
- Business & ML problem definition  
- Dataset scouting + selection  
- Label strategy  
- Feature list  
- Architecture design  
- System diagram    


