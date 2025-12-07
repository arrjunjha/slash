
# README — Sprint 1: System Design  
### Project: Predicting Product Return Risk in Retail Using Classification

## 1. Overview
In retail and e-commerce, product returns contribute significantly to operational and financial losses. This project aims to design a machine learning system that predicts whether an order will be returned or cancelled at the time of purchase. This README documents all deliverables for Sprint 1.

## 2. Business Problem
Retailers lose money on logistics, refunds, and product handling due to returns. Predicting return risk at order time allows interventions such as confirmation prompts, fraud detection, recommendation adjustments, and optimized shipping decisions.

**Business Question:** “Will this item/order be returned?”

## 3. Machine Learning Problem
- **Type:** Supervised Learning — Binary Classification  
- **Target Variable:**  
  - `1` → Returned or Cancelled  
  - `0` → Not Returned  
- **Input Features:** Product details, customer attributes, order context, pricing, quantity, and purchase channel.

The objective is to train a classifier that performs significantly better than simple baselines.

## 4. Dataset Scouting & Selection
Two potential datasets were considered:

### Candidate Dataset 1: Kaggle E-Commerce Returns Dataset
- Contains product, customer, order, and return status  
- Clean return labels  
- Good volume and structure

### Candidate Dataset 2: Online Retail Dataset (UCI)
- Invoice-level retail logs  
- Returned items indicated via negative quantities

### Selected Dataset: Kaggle E-Commerce Returns Dataset  
**Reason:** Clear return flags, strong feature coverage, order-time availability, minimal ambiguity, and suitability for classification.

## 5. Label Definition
```
1 = Returned / Cancelled
0 = Not Returned
```
If return flag exists → map directly.  
If using invoice data → quantity < 0 indicates a return.

## 6. Key Features
- **Product/Order:** Category, price, discount, quantity, vendor ID  
- **Customer:** Customer ID, historical return rate, total orders, region  
- **Context:** Purchase channel, day/time, seasonality, payment mode  

These are known correlates of return behavior.

## 7. High-Level System Architecture
- **Python** for data handling and ML (pandas, sklearn, XGBoost)  
- **FastAPI/Flask** for model serving  
- **HTML/JS** minimal UI  
- **joblib** for model serialization  

Flow:
1. UI collects order details  
2. API receives payload  
3. Model loads & predicts  
4. UI displays return risk  

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

## 9. Team Roles
| Role | Responsibilities |
|------|------------------|
| Data Engineer | Dataset search, cleaning, preprocessing |
| ML Engineer | Feature engineering, model training, evaluation |
| Backend Engineer | API development and integration |
| Frontend/Documentation | UI creation, diagrams, documentation |

## 10. Completed Deliverables (Sprint 1)
- Business & ML problem definition  
- Dataset scouting + selection  
- Label strategy  
- Feature list  
- Architecture design  
- System diagram  
- Team responsibility allocation  

## 11. Next Steps (Sprint 2)
- EDA  
- Feature processing  
- Model training  
- Baseline comparison  
- Evaluation (Precision, Recall, F1, ROC-AUC)  
- Model optimization  

