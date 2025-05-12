# 🚗 Car Price Analysis & Prediction – India Dataset

This project focuses on analyzing car data from India and predicting vehicle prices using machine learning and deep learning models. The study involves comprehensive data processing, visualization, and modeling for both regression and classification tasks.

---

## 📁 Dataset

- **Source:** `car_dataset_india.csv`
- Contains vehicle-related attributes including year, price, fuel type, transmission, engine size, mileage, and more.

---

## 🧹 A. Data Processing & Preparation

- Loaded the dataset using pandas and performed initial inspection (`df.head()`, `df.info()`, `df.describe()`).
- Checked for and addressed **missing values**.
- **Label Encoding** was applied to transform categorical variables to numeric.
- Data was split into **training (80%) and testing (20%)** sets.
- **StandardScaler** was used for feature scaling and normalization.

---

## 📊 B. Data Visualization

A total of **10 different plots** were created to understand data distribution and feature relationships:

1. **Vehicle Price Distribution** – Histogram & KDE
2. **Vehicles Per Year** – Count plot by manufacturing year
3. **Fuel Type Distribution** – CNG, Diesel, Petrol, Electric
4. **Transmission Type Distribution** – Automatic vs Manual
5. **Engine Capacity Distribution** – Histogram
6. **Mileage Distribution** – Histogram
7. **Engine Capacity vs. Price** – Scatter plot
8. **Year vs. Vehicle Price** – Box plot
9. **Fuel Type vs. Price** – Box plot
10. **Service Cost Distribution** – Histogram

---

## 🤖 C. Machine Learning Models

### 🧮 Regression Models – For Predicting Car Price

- **Models Used:**
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Support Vector Regressor (SVR)

- **Evaluated On:**
  - R² Score
  - Training Time (seconds)

### 🧠 Classification Models – For Predicting Price Category

- **Models Used:**
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes

- **Evaluated On:**
  - Accuracy Score
  - Training Time (seconds)

---

## 🧠 D. Deep Learning Models (Regression)

Two deep learning models were created using **TensorFlow** and **Keras**:

### 🔹 Model 1

- Architecture: 64 → 128 → 64 → 1
- Activation: ReLU
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error (MSE)

### 🔹 Model 2

- Architecture: 128 → 256 → 128 → 1
- Activation: ReLU
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error (MSE)

### 🔧 Training Parameters

- Epochs: 50
- Batch Size: 32

### 📊 Results

| Model  | Loss                | MAE        |
|--------|---------------------|------------|
| Model 1| 789,820,538,880.0   | 763,534.63 |
| Model 2| 786,706,989,056.0   | 761,921.25 |

- MAE curves and training vs validation errors were plotted for both models.

---

## 🛠️ Tools & Libraries Used

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow, Keras
- Jupyter Notebook

---

## 📌 Summary

This project demonstrated how vehicle data can be leveraged to predict prices with good accuracy. Machine learning models like **Random Forest** and **Gradient Boosting** performed well for regression and classification, while deep learning models provided comparable accuracy with more training time.

---


