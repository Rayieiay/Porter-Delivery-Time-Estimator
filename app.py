import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import datetime
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Porter Delivery Time Estimator",
    page_icon="🚚",
    layout="centered"
)

# Load model function
@st.cache(allow_output_mutation=True)
def load_model():
    """Load the trained model and feature info"""
    try:
        with open('porter_delivery_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
            
        return model, feature_info
    except FileNotFoundError:
        st.error("❌ Model files not found!")
        return None, None

def main():
    st.title("🚚 Porter Delivery Time Estimator")
    st.markdown("Prediksi waktu pengiriman berdasarkan data pesanan")
    st.markdown("---")
    
    # Load model
    model, feature_info = load_model()
    
    if model is None or feature_info is None:
        st.stop()
    
    # Input form
    st.subheader("📝 Input Data Pesanan")
    
    # Basic Information
    st.markdown("**📍 Informasi Dasar**")
    market_id = st.number_input("Market ID", min_value=1, max_value=10, value=2)
    store_category = st.selectbox(
        "Kategori Toko", 
        ['american', 'mexican', 'italian', 'chinese', 'indian', 'unknown']
    )
    order_protocol = st.selectbox("Protocol Pesanan", [1.0, 2.0, 3.0])
    
    # Order Details
    st.markdown("**🛒 Detail Pesanan**")
    total_items = st.number_input("Total Items", min_value=1, max_value=50, value=5)
    subtotal = st.number_input("Subtotal (dalam cent)", min_value=100, max_value=50000, value=3500)
    num_distinct_items = st.number_input("Jumlah Item Berbeda", min_value=1, max_value=30, value=4)
    min_item_price = st.number_input("Harga Item Minimum", min_value=50, max_value=10000, value=500)
    max_item_price = st.number_input("Harga Item Maximum", min_value=100, max_value=15000, value=1000)
    
    # Partner Information
    st.markdown("**👥 Informasi Partner**")
    total_onshift_partners = st.number_input("Total Partner On-shift", min_value=1, max_value=100, value=20)
    total_busy_partners = st.number_input("Total Partner Sibuk", min_value=0, max_value=50, value=10)
    total_outstanding_orders = st.number_input("Total Pesanan Outstanding", min_value=0, max_value=200, value=15)
    
    # Time Information
    st.markdown("**⏰ Waktu Pesanan**")
    created_hour = st.slider("Jam Pemesanan (0-23)", 0, 23, 13)
    
    day_options = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    selected_day = st.selectbox("Hari dalam Minggu", day_options)
    created_dayofweek = day_options.index(selected_day)
    created_is_weekend = 1 if created_dayofweek >= 5 else 0
    
    # Predict button
    st.markdown("---")
    
    if st.button("🔮 Prediksi Waktu Pengiriman"):
        # Prepare input data
        input_data = {
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
        }
        
        # Create DataFrame
        df_input = pd.DataFrame(input_data)
        
        # Make prediction
        try:
            prediction = model.predict(df_input)[0]
            
            # Display results
            st.success("✅ Prediksi Berhasil!")
            
            # Main result
            st.markdown(f"""
            <div style='background-color: #e1f5fe; padding: 20px; border-radius: 10px; text-align: center;'>
                <h2 style='color: #0277bd;'>🎯 Hasil Prediksi</h2>
                <h1 style='color: #01579b;'>{prediction:.1f} menit</h1>
                <p style='color: #0288d1;'>≈ {prediction/60:.1f} jam</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Additional information
            if prediction <= 30:
                category = "🟢 Cepat"
                st.success(f"Kategori: {category}")
            elif prediction <= 60:
                category = "🟡 Normal"
                st.warning(f"Kategori: {category}")
            else:
                category = "🔴 Lambat"
                st.error(f"Kategori: {category}")
            
            # Calculate ETA
            now = datetime.datetime.now()
            estimated_arrival = now + datetime.timedelta(minutes=prediction)
            st.info(f"🕐 Perkiraan tiba: {estimated_arrival.strftime('%H:%M')}")
            
            # Insights
            st.markdown("**💡 Insights:**")
            
            if total_busy_partners > total_onshift_partners * 0.7:
                st.warning("⚠️ Banyak partner yang sibuk, mungkin akan mempengaruhi waktu pengiriman")
            
            if total_outstanding_orders > 50:
                st.warning("📈 Pesanan outstanding tinggi, kemungkinan ada delay")
            
            if created_is_weekend:
                st.info("🎉 Pesanan di akhir pekan, pola pengiriman mungkin berbeda")
            
            if created_hour < 8 or created_hour > 22:
                st.info("🌙 Pesanan di luar jam normal, waktu pengiriman mungkin lebih lama")
            
            # Summary
            st.markdown("---")
            st.markdown(f"""
            **📋 Ringkasan:**
            - **Waktu pengiriman:** {prediction:.1f} menit ({prediction/60:.2f} jam)
            - **Kategori:** {category}
            - **Estimasi tiba:** {estimated_arrival.strftime('%H:%M')}
            - **Hari:** {selected_day} ({'Akhir pekan' if created_is_weekend else 'Hari kerja'})
            - **Jam pemesanan:** {created_hour}:00
            """)
            
        except Exception as e:
            st.error(f"❌ Error dalam prediksi: {str(e)}")
            st.error("Pastikan semua input sudah diisi dengan benar.")
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ Informasi")
    
    st.info("""
    **Model yang Digunakan:**
    - Linear Regression dengan Pipeline
    - Preprocessing: StandardScaler + OneHotEncoder
    - 14 fitur input (13 numerical + 1 categorical)
    """)
    
    st.info("""
    **Cara Penggunaan:**
    1. Isi semua field input yang tersedia
    2. Pastikan nilai-nilai masuk akal dan realistis
    3. Klik tombol "Prediksi Waktu Pengiriman"
    4. Lihat hasil prediksi dan insights yang diberikan
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("🚚 **Porter Delivery Time Estimator** | Dikembangkan oleh Kelompok 4")

if __name__ == "__main__":
    main()