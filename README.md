# Bank Customer Churn Prediction using ANN

This project implements an Artificial Neural Network (AN) to predict whether a bank customer is likely to leave the bank (churn) based on their demographics and financial behavior. The model is built using **TensorFlow/Keras** and trained on the `Churn_Modelling.csv` dataset.

## 📊 Dataset Overview

The dataset contains details of bank customers, including:

* **Features:** Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, and Estimated Salary.
* **Target:** `Exited` (1 if the customer left the bank, 0 if they stayed).

---

## 🏗️ Project Workflow

### 1. Data Cleaning & Exploration

* **Feature Selection:** Dropped non-predictive columns: `RowNumber`, `CustomerId`, and `Surname`.
* **Exploratory Data Analysis (EDA):** Inspected data distributions using `df.info()`, `df.shape`, and checked for duplicates.
* **Target Balance:** Analyzed the distribution of the `Exited` class to understand the label spread.

### 2. Preprocessing

* **Categorical Encoding:** Applied **One-Hot Encoding** to `Geography` and `Gender` using `pd.get_dummies()`. The "first category" was dropped to prevent the Dummy Variable Trap (multicollinearity).
* **Data Splitting:** Divided the data into an 80/20 Train-Test split.
* **Feature Scaling:** Utilized `StandardScaler` on the features. This is a critical step for Neural Networks to ensure the **Gradient Descent** optimizer converges efficiently.

### 3. Model Architecture

A Sequential ANN was constructed with the following layers:

* **Input/Hidden Layer:** 6 neurons with **ReLU** activation.
* **Output Layer:** 1 neuron with **Sigmoid** activation (ideal for binary classification as it outputs a probability between 0 and 1).

### 4. Compilation & Training

* **Optimizer:** `Adam` (Adaptive Moment Estimation).
* **Loss Function:** `binary_crossentropy`.
* **Epochs:** 50.
* **Validation:** 20% of the training data was used for real-time validation during the training process.

---

## 📈 Performance Visualization

The notebook includes detailed plots for:

* **Training vs. Validation Accuracy:** To monitor learning progress.
* **Training vs. Validation Loss:** To check for potential overfitting or underfitting.

---

## 🛠️ Tech Stack

* **Data Analysis:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `Scikit-Learn`
* **Deep Learning:** `TensorFlow`, `Keras`

---

## 🚀 How to Run

1. **Clone the repo:**
```bash
git clone https://github.com/your-username/churn-ann-prediction.git

```


2. **Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

```


3. **Run the Notebook:** Open `Churn_Prediction.ipynb` in Google Colab or Jupyter and ensure `Churn_Modelling.csv` is in the same directory.


