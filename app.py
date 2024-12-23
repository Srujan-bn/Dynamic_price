import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os
import numpy as np
from datetime import datetime, timedelta
import bcrypt
import json
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Ensure required directories exist
def ensure_directory_structure():
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists("credentials.json"):
        with open("credentials.json", "w") as f:
            json.dump({"users": {}}, f)

ensure_directory_structure()

# Database setup for complaints
def ensure_database():
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            complaint TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

ensure_database()

# Store complaints in the database
def store_complaint(username, complaint_text):
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO complaints (username, complaint) VALUES (?, ?)", (username, complaint_text))
    conn.commit()
    conn.close()
    st.success("Complaint recorded successfully!")

# View complaints functionality
def view_complaints():
    st.header("View Complaints")
    conn = sqlite3.connect("complaints.db")
    complaints = pd.read_sql_query("SELECT * FROM complaints ORDER BY timestamp DESC", conn)
    conn.close()

    if complaints.empty:
        st.write("No complaints found.")
    else:
        st.dataframe(complaints)
        csv = complaints.to_csv(index=False)
        st.download_button("Download Complaints as CSV", csv, "complaints.csv", "text/csv")

# Authentication Helper Functions
CREDENTIALS_FILE = "credentials.json"

def load_credentials():
    with open(CREDENTIALS_FILE, "r") as f:
        return json.load(f)

def save_credentials(credentials):
    with open(CREDENTIALS_FILE, "w") as f:
        json.dump(credentials, f)

def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

def authenticate_user(username, password):
    credentials = load_credentials()
    if username in credentials["users"]:
        return verify_password(password, credentials["users"][username])
    return False

def register_user(username, password):
    credentials = load_credentials()
    if username in credentials["users"]:
        return False  # User already exists
    credentials["users"][username] = hash_password(password)
    save_credentials(credentials)
    return True

# Machine Learning Helper Functions
def train_dynamic_pricing_model(data):
    features = ['price', 'remaining_stock', 'quantity_sold', 'num_customers_visited']
    target = 'optimal_price'
    
    if target not in data.columns:
        data[target] = data['price'] * (1 + np.random.uniform(-0.2, 0.2, len(data)))

    X = data[features]
    y = data[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    with open("models/dynamic_pricing_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler

def predict_optimal_price(model, scaler, input_data):
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)

# Streamlit App
def login_page():
    st.title("Login")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_button = st.button("Login")

        if login_button:
            if authenticate_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid username or password.")

    with tab2:
        username = st.text_input("New Username", key="register_username")
        password = st.text_input("New Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
        register_button = st.button("Register")

        if register_button:
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif register_user(username, password):
                st.success("Registration successful! You can now log in.")
            else:
                st.error("Username already exists.")

def main_app():
    st.title("Dynamic Pricing Optimization for Retail Shops")

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.experimental_rerun()

    # Step 1: Dataset upload
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file with the required columns", type="csv")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            data['date'] = pd.to_datetime(data['date'])

            required_columns = ['item_name', 'date', 'price', 'remaining_stock', 'quantity_sold', 'num_customers_visited']
            if not all(col in data.columns for col in required_columns):
                st.error(f"Dataset must contain the following columns: {', '.join(required_columns)}")
                return

            st.success("Dataset uploaded successfully!")

            # Train model if not already trained
            if 'model' not in st.session_state:
                if st.button("Train Model"):
                    st.info("Training the model, please wait...")
                    model, scaler = train_dynamic_pricing_model(data)
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.success("Model trained successfully!")

            # Step 2: Input fields for predictions
            st.header("Provide Current Inputs")
            item_name = st.selectbox("Select Item", data['item_name'].unique(), key="item_name")
            current_price = st.number_input("Current Price", min_value=0.0, format="%.2f", key="current_price")
            current_inventory = st.number_input("Current Inventory Quantity", min_value=0, key="current_inventory")
            quantity_sold = st.number_input("Quantity Sold (Optional)", min_value=0, value=12, key="quantity_sold")
            num_customers_visited = st.number_input("Number of Customers Visited (Optional)", min_value=0, value=20, key="num_customers_visited")

            if st.button("Generate Recommendations"):
                if 'model' in st.session_state:
                    model = st.session_state.model
                    scaler = st.session_state.scaler

                    input_data = pd.DataFrame({
                        'price': [current_price],
                        'remaining_stock': [current_inventory],
                        'quantity_sold': [quantity_sold],
                        'num_customers_visited': [num_customers_visited],
                    })

                    optimal_price = predict_optimal_price(model, scaler, input_data)
                    optimal_price = np.clip(optimal_price, current_price * 0.8, current_price * 1.2)

                    # Separate Profit and Loss
                    profit_loss_percentage = (optimal_price - current_price) / current_price * 100
                    profit = profit_loss_percentage[0] if profit_loss_percentage[0] > 0 else 0
                    loss = abs(profit_loss_percentage[0]) if profit_loss_percentage[0] < 0 else 0

                    st.write(f"Optimal Price for {item_name}: ₹{optimal_price[0]:.2f}")
                    if profit > 0:
                        st.success(f"Profit: {profit:.2f}%")
                    if loss > 0:
                        st.error(f"Loss: {loss:.2f}%")

                    # Graph: Profit/Loss Percentage
                    days = range(30)
                    profit_loss_values = [profit_loss_percentage[0]] * 30
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(days, profit_loss_values, label="Profit/Loss (%)", color="green" if profit > 0 else "red")
                    ax.set_xlabel("Days")
                    ax.set_ylabel("Profit/Loss (%)")
                    ax.legend()
                    st.pyplot(fig)

            # Graph: Past 6 Months Sales
            st.header("Sales Trends (Past 6 Months)")
            six_months_ago = datetime.now() - timedelta(days=180)
            past_sales_data = data[data['date'] > six_months_ago]

            if not past_sales_data.empty:
                monthly_sales = past_sales_data.groupby(past_sales_data['date'].dt.to_period('M'))['quantity_sold'].sum()
                fig, ax = plt.subplots(figsize=(10, 5))
                monthly_sales.plot(kind='bar', ax=ax, color="blue")
                ax.set_title("Sales Over the Past 6 Months")
                ax.set_xlabel("Month")
                ax.set_ylabel("Quantity Sold")
                st.pyplot(fig)
            else:
                st.write("No sales data available for the past 6 months.")

        except Exception as e:
            st.error(f"Error processing the file: {e}")

    # Complaint Box Feature
    st.header("Complaint Box")
    complaint_text = st.text_area("Please describe the issue you're facing:")

    if st.button("Send Complaint"):
        if complaint_text:
            username = st.session_state.get("username", "Anonymous")
            store_complaint(username, complaint_text)
        else:
            st.error("Please enter a complaint before sending.")

    # Admin view for complaints
    if st.sidebar.checkbox("Admin: View Complaints"):
        if st.session_state.get("username") == "admin":  # Replace with your admin username
            view_complaints()
        else:
            st.error("Access denied. Admins only.")

def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()
