# 🛒 E-Commerce Customer Segmentation & Profiling (K-Means Clustering)

## Project Overview
Customer segmentation helps businesses understand different customer groups and tailor marketing strategies accordingly. This project analyzes customer transaction patterns to categorize them into 3 distinct business clusters:
- **VIP Customers:** High-value customers with high transaction frequency and monetary contribution.
- **Regular Customers:** Consistent shoppers with average frequency and spend.
- **Churn Risk Customers:** Inactive customers who haven't made recent purchases.

## Installation
1. Clone the repository
    ```bash
    git clone https://github.com/hassantaufiqurrahman/E-Commerce-Customer-Segmentation.git
    cd E-Commerce-Customer-Segmentation
    ```

2. Create and activate virtual environment
    ```bash
    py -3.11 -m venv .venv
    .venv\Scripts\activate
    ```

3. Upgrade pip and install dependencies
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. Run the Streamlit application
    ```bash
    streamlit run app.py
    ```