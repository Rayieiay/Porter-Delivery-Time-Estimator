import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Porter ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.prediction-result {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    text-align: center;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Load model with caching for older versions
@st.cache(allow_output_mutation=True)
def load_model_and_features():
    """Load model and feature information"""
    try:
        with open('porter_delivery_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
            
        return model, feature_info
    except FileNotFoundError:
        return None, None

def create_sample_data():
    """Create sample data for testing"""
    return {
        'market_id': 2,
        'store_primary_category': 'american',
        'order_protocol': 1.0,
        'total_items': 5,
        'subtotal': 3500,
        'num_distinct_items': 4,
        'min_item_price': 500,
        'max_item_price': 1000,
        'total_onshift_partners': 20,
        'total_busy_partners': 10,
        'total_outstanding_orders': 15,
        'created_hour': 13,
        'created_dayofweek': 2,
        'created_is_weekend': 0
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🚚 Porter Delivery ML Dashboard</h1>', unsafe_allow_html=True)
    
    # Load model
    model, feature_info = load_model_and_features()
    
    if model is None:
        st.error("❌ Model tidak ditemukan! Pastikan file model ada di direktori yang sama.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Pilih Halaman",
        ["🔮 Prediksi", "📊 Batch Prediction", "📈 Model Analysis", "ℹ️ About"]
    )
    
    if page == "🔮 Prediksi":
        prediction_page(model, feature_info)
    elif page == "📊 Batch Prediction":
        batch_prediction_page(model, feature_info)
    elif page == "📈 Model Analysis":
        model_analysis_page(model, feature_info)
    else:
        about_page()

def prediction_page(model, feature_info):
    """Single prediction page"""
    st.header("🔮 Prediksi Waktu Pengiriman")
    
    # Quick prediction buttons
    st.subheader("⚡ Quick Predictions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🍔 Fast Food Order"):
            predict_quick_scenario(model, 'fast_food')
    
    with col2:
        if st.button("🍕 Regular Order"):
            predict_quick_scenario(model, 'regular')
    
    with col3:
        if st.button("🥘 Large Order"):
            predict_quick_scenario(model, 'large')
    
    st.markdown("---")
    
    # Manual input section
    st.subheader("🎛️ Custom Prediction")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📍 Location & Store**")
            market_id = st.number_input("Market ID", 1, 10, 2)
            store_category = st.selectbox(
                "Store Category",
                ['american', 'mexican', 'italian', 'chinese', 'indian', 'unknown']
            )
            order_protocol = st.selectbox("Order Protocol", [1.0, 2.0, 3.0])
        
        with col2:
            st.markdown("**🛒 Order Details**")
            total_items = st.number_input("Total Items", 1, 50, 5)
            subtotal = st.number_input("Subtotal", 100, 50000, 3500)
            num_distinct_items = st.number_input("Distinct Items", 1, 30, 4)
            min_item_price = st.number_input("Min Item Price", 50, 10000, 500)
            max_item_price = st.number_input("Max Item Price", 100, 15000, 1000)
        
        with col3:
            st.markdown("**👥 Partner Info**")
            total_onshift_partners = st.number_input("Onshift Partners", 1, 100, 20)
            total_busy_partners = st.number_input("Busy Partners", 0, 50, 10)
            total_outstanding_orders = st.number_input("Outstanding Orders", 0, 200, 15)
            
            st.markdown("**⏰ Time Info**")
            created_hour = st.slider("Hour of Day", 0, 23, 13)
            created_dayofweek = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
        
        submitted = st.form_submit_button("🚀 Predict Delivery Time")
        
        if submitted:
            # Prepare data
            created_is_weekend = 1 if created_dayofweek >= 5 else 0
            
            input_data = pd.DataFrame({
                'market_id': [market_id],
                'store_primary_category': [store_category],
                'order_protocol': [order_protocol],
                'total_items': [total_items],
                'subtotal': [subtotal],
                'num_distinct_items': [num_distinct_items],
                'min_item_price': [min_item_price],
                'max_item_price': [max_item_price],
                'total_onshift_partners': [total_onshift_partners],
                'total_busy_partners': [total_busy_partners],
                'total_outstanding_orders': [total_outstanding_orders],
                'created_hour': [created_hour],
                'created_dayofweek': [created_dayofweek],
                'created_is_weekend': [created_is_weekend]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            display_prediction_result(prediction, input_data.iloc[0])

def predict_quick_scenario(model, scenario_type):
    """Quick prediction for predefined scenarios"""
    scenarios = {
        'fast_food': {
            'market_id': 1, 'store_primary_category': 'american', 'order_protocol': 3.0,
            'total_items': 2, 'subtotal': 1500, 'num_distinct_items': 2,
            'min_item_price': 500, 'max_item_price': 1000, 'total_onshift_partners': 25,
            'total_busy_partners': 5, 'total_outstanding_orders': 10,
            'created_hour': 12, 'created_dayofweek': 2, 'created_is_weekend': 0
        },
        'regular': {
            'market_id': 2, 'store_primary_category': 'italian', 'order_protocol': 2.0,
            'total_items': 5, 'subtotal': 3500, 'num_distinct_items': 4,
            'min_item_price': 400, 'max_item_price': 1200, 'total_onshift_partners': 20,
            'total_busy_partners': 12, 'total_outstanding_orders': 18,
            'created_hour': 19, 'created_dayofweek': 4, 'created_is_weekend': 0
        },
        'large': {
            'market_id': 3, 'store_primary_category': 'chinese', 'order_protocol': 1.0,
            'total_items': 12, 'subtotal': 8500, 'num_distinct_items': 8,
            'min_item_price': 300, 'max_item_price': 2000, 'total_onshift_partners': 15,
            'total_busy_partners': 18, 'total_outstanding_orders': 35,
            'created_hour': 20, 'created_dayofweek': 5, 'created_is_weekend': 1
        }
    }
    
    data = scenarios[scenario_type]
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    
    st.success(f"⚡ {scenario_type.title()} order prediction: **{prediction:.1f} minutes**")

def display_prediction_result(prediction, input_data):
    """Display prediction results with visualization"""
    
    # Main result
    st.markdown(f"""
    <div class="prediction-result">
        <h2>🎯 Prediction Result</h2>
        <h1>{prediction:.1f} minutes</h1>
        <p>≈ {prediction/60:.1f} hours</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category = "🟢 Fast" if prediction <= 30 else "🟡 Normal" if prediction <= 60 else "🔴 Slow"
        st.metric("Category", category)
    
    with col2:
        eta = datetime.now() + timedelta(minutes=prediction)
        st.metric("ETA", eta.strftime("%H:%M"))
    
    with col3:
        confidence = "High" if 20 <= prediction <= 80 else "Medium"
        st.metric("Confidence", confidence)
    
    with col4:
        rush_hour = "Yes" if input_data['created_hour'] in [11, 12, 13, 18, 19, 20] else "No"
        st.metric("Rush Hour", rush_hour)
    
    # Visualization
    create_prediction_chart(prediction)

def create_prediction_chart(prediction):
    """Create a gauge chart for prediction"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Delivery Time (minutes)"},
        delta = {'reference': 45},
        gauge = {
            'axis': {'range': [None, 120]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 120], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def batch_prediction_page(model, feature_info):
    """Batch prediction page"""
    st.header("📊 Batch Predictions")
    
    st.info("Upload a CSV file with the required columns to get predictions for multiple orders.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Check required columns
            required_cols = feature_info['feature_columns']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                if st.button("🚀 Generate Predictions"):
                    # Make predictions
                    predictions = model.predict(df[required_cols])
                    df['predicted_delivery_minutes'] = predictions
                    df['predicted_delivery_hours'] = predictions / 60
                    
                    # Show results
                    st.subheader("📈 Prediction Results")
                    st.dataframe(df)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Average Delivery Time", f"{predictions.mean():.1f} min")
                    
                    with col2:
                        st.metric("Median Delivery Time", f"{np.median(predictions):.1f} min")
                    
                    with col3:
                        st.metric("Max Delivery Time", f"{predictions.max():.1f} min")
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Sample data
    st.subheader("📝 Sample Data Format")
    sample_data = create_sample_data()
    sample_df = pd.DataFrame([sample_data])
    st.dataframe(sample_df)

def model_analysis_page(model, feature_info):
    """Model analysis page"""
    st.header("📈 Model Analysis")
    
    # Model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Details")
        st.info("""
        **Model Type:** Linear Regression Pipeline
        **Preprocessing:** StandardScaler + OneHotEncoder
        **Features:** 14 total (13 numerical + 1 categorical)
        """)
        
        st.subheader("📊 Feature List")
        st.write("**Numerical Features:**")
        for feature in feature_info['numerical_features']:
            st.write(f"- {feature}")
        
        st.write("**Categorical Features:**")
        for feature in feature_info['categorical_features']:
            st.write(f"- {feature}")
    
    with col2:
        st.subheader("🎯 Model Performance")
        st.warning("""
        **Note:** This is a demo model trained on synthetic data.
        For production use, train with real historical data.
        """)
        
        st.subheader("💡 Model Insights")
        st.info("""
        **Key Factors Affecting Delivery Time:**
        - Number of busy partners
        - Total outstanding orders
        - Time of day (rush hours)
        - Weekend vs weekday
        - Order size and complexity
        """)

def about_page():
    """About page"""
    st.header("ℹ️ About Porter Delivery Time Estimator")
    
    st.markdown("""
    ## 🎯 Purpose
    This application predicts delivery times for Porter orders based on various factors including:
    - Order characteristics (items, price, etc.)
    - Partner availability
    - Time factors
    - Market conditions
    
    ## 🔧 Technology Stack
    - **Frontend:** Streamlit
    - **ML Model:** Scikit-learn Linear Regression
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Plotly
    
    ## 📊 Model Features
    The model uses 14 features to predict delivery time:
    - Market and store information
    - Order details (items, prices)
    - Partner availability
    - Time-based features
    
    ## ⚠️ Important Notes
    - This is a demonstration model trained on synthetic data
    - For production use, retrain with real historical delivery data
    - Predictions should be validated against actual delivery times
    
    ## 👥 Team
    Developed by Kelompok 4 for Porter Delivery Time Estimation Project
    """)

if __name__ == "__main__":
    main()