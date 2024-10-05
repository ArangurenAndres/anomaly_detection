# **Anomaly Detection in Marketplace User Activity Using LSTM and Transformers**

## **Project Overview**

This project focuses on building a machine learning framework for detecting anomalies in user activities on a marketplace platform (similar to Vinted). The goal is to identify abnormal user behaviors such as fraudulent transactions, unusual purchase activity, or any behavior that could indicate potential harm within the system.

We leverage two advanced models for anomaly detection:
- **LSTM (Long Short-Term Memory)**: A powerful type of recurrent neural network (RNN) for sequential data.
- **Transformers**: A modern architecture that excels at capturing long-range dependencies in sequential data.

The **Online Retail Dataset** from the UCI Machine Learning Repository serves as the foundation for this project. This dataset contains transactions for a UK-based online retail store and is suitable for detecting unusual patterns in user purchases.

---

## **Dataset**

The dataset used is the **Online Retail Dataset** from the UCI Machine Learning Repository. It contains transactional data from a UK-based online retail store.

### **Dataset Features**:
- **InvoiceNo**: Invoice number, uniquely identifying each transaction.
- **StockCode**: Product code of the items involved in the transaction.
- **Description**: Description of the product.
- **Quantity**: The quantity of the product bought.
- **InvoiceDate**: The date and time of the transaction.
- **UnitPrice**: Price of the product per unit.
- **CustomerID**: Unique ID for each customer.
- **Country**: The country from which the transaction was made.

### **Download the Dataset:**

You can download the dataset using the following command:

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx -P data/


