import streamlit as st
from ml_app import predict_from_partial 

st.set_page_config(
    page_title="Porter ETA – Lite",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not hasattr(st, "divider"):
    def _divider(*, width="stretch"):
        st.write("---")
    st.divider = _divider

def inject_css():
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] { background: transparent; }

        [data-testid="stSidebar"] {
          background: rgba(15,17,26,.96) !important;
          backdrop-filter: blur(6px);
          border-right: 1px solid rgba(255,255,255,.06);
        }

        [data-testid="stAppViewContainer"] {
          background: radial-gradient(1100px 650px at 20% 15%, #101827 0%, #0b1220 40%, #070b15 100%) fixed;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 { color: #e8e9ee !important; letter-spacing: .2px; }
        p, label, .stText, .stMarkdown, .stMetric { color: #cbd5e1 !important; }

        /* Card */
        .card {
          border-radius: 14px;
          padding: 1rem 1.25rem;
          background: linear-gradient(180deg, rgba(17,24,39,.85) 0%, rgba(11,18,32,.85) 100%);
          border: 1px solid rgba(148,163,184,.18);
          box-shadow: 0 10px 24px rgba(2,6,23,.35);
        }

        /* Tombol */
        .stButton>button {
          border-radius: 10px;
          padding: .65rem 1rem;
          font-weight: 600;
          border: 1px solid rgba(255,255,255,.15);
          background: linear-gradient(180deg, #243b55, #141e30);
          color: #eef2ff;
        }
        .stButton>button:hover { filter: brightness(1.08); border-color: rgba(255,255,255,.25); }

        /* Input */
        div[data-baseweb="input"] input, .stSelectbox [data-baseweb="select"] {
          background: rgba(148,163,184,.08);
          color: #e5e7eb;
        }
        label[for] { margin-bottom: .2rem; font-weight: 500; }

        /* HR */
        hr { border: 0; border-top: 1px solid rgba(148,163,184,.25); margin: .75rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

#Header
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:2px;">
      <div style="font-size:28px">⏱️</div>
      <div>
        <h1 style="margin:0;">Porter Delivery Time Estimation</h1>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

#Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    st.caption("Fitur turunan & default dihasilkan otomatis di backend.")
    st.markdown("---")
    st.markdown(
        """
        **Tips input**  
        • `created_dayofweek`: 0=Mon … 6=Sun  
        • `created_hour`: 0–23  
        • Pastikan angka partner & order ≥ 0
        """
    )
    st.markdown("---")

#Form
st.markdown("#### Input Semua Kolom yang Tersedia")

with st.form("lite_form"):
    c1, c2, c3, out = st.columns([1, 1, 1, 1.25])

    with c1:
        order_protocol = st.selectbox("order_protocol", [1, 2, 3, 4, 5], index=0)
        market_id = st.selectbox("market_id", [1, 2, 3, 4, 5, 6], index=0)
        total_items = st.number_input("total_items", min_value=0, value=3, step=1)

    with c2:
        subtotal = st.number_input("subtotal", min_value=0.0, value=150000.0, step=1000.0)
        created_hour = st.number_input("created_hour (0–23)", min_value=0, max_value=23, value=12, step=1)
        created_dayofweek = st.number_input("created_dayofweek (0=Mon..6=Sun)", min_value=0, max_value=6, value=2, step=1)

    with c3:
        total_onshift_partners = st.number_input("total_onshift_partners", min_value=0, value=100, step=1)
        total_busy_partners = st.number_input("total_busy_partners", min_value=0, value=60, step=1)
        total_outstanding_orders = st.number_input("total_outstanding_orders", min_value=0, value=50, step=1)

        submit = st.form_submit_button("🔮 Prediksi ETA")

    #eta
    if submit:
        features = dict(
            order_protocol=order_protocol,
            market_id=market_id,
            total_items=total_items,
            subtotal=subtotal,
            created_hour=created_hour,
            created_dayofweek=created_dayofweek,
            total_onshift_partners=total_onshift_partners,
            total_busy_partners=total_busy_partners,
            total_outstanding_orders=total_outstanding_orders,
        )
        eta = predict_from_partial(features)
        st.session_state["eta_minutes"] = float(eta)
        st.session_state["features"] = features

    #hasil
    with out:
        if "eta_minutes" not in st.session_state:
            st.metric("Estimasi waktu antar (menit)", "—")
            st.caption("Isi form lalu klik **Prediksi ETA**.")
        else:
            st.metric("Estimasi waktu antar (menit)", round(st.session_state["eta_minutes"], 2))

            f = st.session_state["features"]
            top = max(int(f["total_onshift_partners"]), 0)
            tbp = max(int(f["total_busy_partners"]), 0)
            too = max(int(f["total_outstanding_orders"]), 0)
            available = max(top - tbp, 0)
            busy_ratio = (tbp / top) if top > 0 else 0.0
            opp = (too / available) if available > 0 else 0.0
            dsr = (too / top) if top > 0 else 0.0
            api = (float(f["subtotal"]) / max(int(f["total_items"]), 1)) if int(f["total_items"]) > 0 else 0.0

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.caption("**Ringkasan operasional (turunan dari input):**")
            st.markdown(
                f"""
                • **Available partners**: {available}  
                • **Busy ratio**: {busy_ratio:.2f}  
                • **Order / partner**: {opp:.2f}  
                • **Demand/Supply ratio**: {dsr:.2f}  
                • **Avg price / item**: {api:.0f}  
                • **Created (H, DOW)**: {f["created_hour"]}, {f["created_dayofweek"]}
                """
            )
            st.caption("Angka-angka di atas hanya visualisasi; pipeline tetap memakai input Lite.")
        st.markdown("</div>", unsafe_allow_html=True)

#footer
st.markdown("<br/>", unsafe_allow_html=True)
