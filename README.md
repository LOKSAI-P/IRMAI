# IRMAI Coding Challenge  

This repository contains three **graph-based anomaly detection** projects that analyze **stock trades and financial transactions** to identify outliers, missing trades, and potential risks.  

## **Projects Overview**  

### **1. Stock Trade Outlier Analysis**  
- Detects unusual **foreign exchange (FX) trades** using **Z-score analysis**.  
- Uses **NetworkX** to visualize trade relationships.  
- Identifies abnormal trades based on **volume and price deviations**.  
- Provides an interactive **Gradio UI** for trade analysis.  

### **2. Stock Trade Gap Analysis**  
- Analyzes **stock trade data** for **missing trades** and **sudden price jumps**.  
- Uses **Neo4j** to model trade relationships.  
- Highlights trade gaps using **graph-based visualization**.  
- Interactive **Gradio UI** for trade pattern exploration.  

### **3. Failure Mode & Effects Analysis (FMEA) for Financial Transactions**  
- Implements **FMEA** to assess financial transaction risks.  
- Detects anomalies like **fraud, duplicate payments, and large transactions**.  
- Uses **Neo4j** to store and analyze financial transactions.  
- Provides a **Gradio UI** to visualize financial transaction anomalies.  

## **Technologies Used**  
- **Python** – Core programming language  
- **Neo4j** – Graph database for trade/transaction storage  
- **NetworkX** – For graph-based visualization  
- **Matplotlib** – For plotting graphs  
- **Gradio** – UI framework for interactive analysis  
- **Pandas & NumPy** – Data processing and analysis  

## **Installation & Usage**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/LOKSAI-P/IRMAI_Coding-Challenge-.git
cd IRMAI_Coding-Challenge-
