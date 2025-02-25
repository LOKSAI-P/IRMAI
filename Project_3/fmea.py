import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from fpdf import FPDF
import tempfile

# Neo4j connection details
URL = "bolt://localhost:7687"  
USERNAME = "neo4j"
PASSWORD = "password"

# Connect to Neo4j
driver = GraphDatabase.driver(URL, auth=(USERNAME, PASSWORD))

# Load transaction data from CSV file 
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Generate sample transactions with anomalies 
def generate_synthetic_data(num_trades=20):
    np.random.seed(42)  # For reproducibility
    data = {
        "transaction_id": range(1, num_trades + 1),
        "account_from": np.random.choice(["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"], num_trades),
        "account_to": np.random.choice(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"], num_trades),
        "amount": np.random.randint(-5000, 10000, num_trades).tolist(),  # Allow negative amounts for refunds
        "type": np.random.choice(["transfer", "credit", "refund"], num_trades),
        "timestamp": pd.date_range("2024-02-01", periods=num_trades, freq="h")
    }

    # Introduce some gaps (missing volume and price)
    for i in range(num_trades):
        if np.random.rand() < 0.1:  # 10% chance to introduce a gap
            data["amount"][i] = 0  # Set amount to zero to simulate missing data

    return pd.DataFrame(data)


try:
    df = load_data_from_csv("transactions.csv") 
except FileNotFoundError:
    df = generate_synthetic_data()

# Define failure modes
large_amount_threshold = 3000
df["large_amount"] = df["amount"] > large_amount_threshold
df["zero_amount"] = df["amount"] == 0
df["duplicate"] = df.duplicated(subset=["account_from", "account_to", "amount", "timestamp"], keep=False)
df["failure_mode"] = df["large_amount"] | df["zero_amount"] | df["duplicate"]

# Assign severity, occurrence, and detection ratings
def calculate_rpn(severity, occurrence, detection):
    return severity * occurrence * detection

# Example ratings for each failure mode
failure_modes = {
    "large_amount": {"severity": 8, "occurrence": 4, "detection": 6},
    "zero_amount": {"severity": 7, "occurrence": 3, "detection": 8},
    "duplicate": {"severity": 9, "occurrence": 2, "detection": 7}
}

# Calculate RPN for each failure mode
rpn_values = {}
for mode, ratings in failure_modes.items():
    rpn = calculate_rpn(ratings["severity"], ratings["occurrence"], ratings["detection"])
    rpn_values[mode] = rpn

# Prioritize failure modes based on RPN
prioritized_modes = sorted(rpn_values.items(), key=lambda x: x[1], reverse=True)

# Display prioritized failure modes
print("Prioritized Failure Modes:")
for mode, rpn in prioritized_modes:
    print(f"Failure Mode: {mode}, RPN: {rpn}")

# Insert transactions into Neo4j
def insert_transaction(tx, transaction):
    query = """
    CREATE (t:Transaction {transaction_id: $transaction_id, account_from: $account_from,
                           account_to: $account_to, amount: $amount, type: $type, timestamp: $timestamp})
    """
    tx.run(query, **transaction)

with driver.session() as session:
    for _, row in df.iterrows():
        session.write_transaction(insert_transaction, row.to_dict())

print("Inserted transactions into Neo4j!")

# Create relationships in Neo4j
def create_relationships(tx):
    query = """
    MATCH (t1:Transaction), (t2:Transaction) 
    WHERE t1.timestamp < t2.timestamp
    CREATE (t1)-[:NEXT_TRANSACTION]->(t2)
    """
    tx.run(query)

with driver.session() as session:
    session.write_transaction(create_relationships)

print("Created transaction relationships in Neo4j!")

# Function to plot the financial transactions graph
def plot_graph():
    G = nx.DiGraph()

    for _, row in df.iterrows():
        color = "red" if row.get("failure_mode", False) else "green"
        G.add_node(row["transaction_id"], color=color, label=f"{row['transaction_id']}")

    for i in range(len(df) - 1):
        G.add_edge(df.loc[i, "transaction_id"], df.loc[i+1, "transaction_id"])

    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]["color"] for n in G.nodes]

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray", 
            node_size=1000, font_size=12, font_weight="bold", arrows=True)

    labels = {n: G.nodes[n]["label"] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title("Financial Transactions with Failure Modes")
    
    # Save image for Streamlit output
    plt.savefig("graph.png")
    
    return "graph.png"

# Function to return DataFrame for table display
def get_transaction_data():
    return df[["transaction_id", "account_from", "account_to", "amount", "type", 
                "timestamp", "failure_mode"]]

# Function to generate PDF report summarizing findings
def generate_report():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Financial Transactions Analysis Report", ln=True, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Total Transactions Analyzed: {len(df)}", ln=True)
    pdf.cell(0, 8, f"Total Anomalies Detected: {df['failure_mode'].sum()}", ln=True)
    pdf.cell(0, 8, f"Large Amount Transactions (> ${large_amount_threshold}): {df['large_amount'].sum()}", ln=True)
    pdf.cell(0, 8, f"Zero Amount Transactions: {df['zero_amount'].sum()}", ln=True)
    pdf.cell(0, 8, f"Duplicate Transactions Detected: {df['duplicate'].sum()}", ln=True)
    pdf.ln(8)

    # Add prioritized failure modes to the report
    pdf.set_font("Arial", "B", 9)
    pdf.cell(0, 8, "Prioritized Failure Modes:", ln=True)
    pdf.set_font("Arial", "", 9)
    for mode, rpn in prioritized_modes:
        pdf.cell(0, 8, f"{mode}: RPN = {rpn}", ln=True)
    
    pdf.ln(8)

    pdf.set_font("Arial", "B", 9)
    pdf.cell(0, 8, "Transaction Data Table:", ln=True)
    pdf.set_font("Arial", "", 9)  
    column_width = 25  
    headers = df.columns.tolist()
    
    for header in headers:
        pdf.cell(column_width, 8, header, border=1, align="C")
    pdf.ln()

    for _, row in df.iterrows():
        for value in row:
            pdf.cell(column_width, 8, str(value), border=1, align="C")
        pdf.ln()

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name 
    pdf.output(pdf_path) 
    return pdf_path

# Streamlit UI setup
st.title("💰 Financial Transactions Analysis")

col1, col2 = st.columns(2)
with col1:
    if st.button("Generate Transaction Graph"):
        plot_graph()
        st.image("graph.png")

with col2:
    if st.button("Generate Report"):
        report_path = generate_report()
        with open(report_path, "rb") as f:
            st.download_button("Download Report", f, file_name="report.pdf")

st.subheader("📊 Transaction Data Table")
st.write(get_transaction_data())
