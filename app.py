import sys
import types

# ──────────────────────────────────────────────────────────────────
# 1. Compatibility Shims for Scikit-Learn 1.8.0 / Imbalanced-Learn
# ──────────────────────────────────────────────────────────────────

# Patch sklearn.utils._tags BEFORE any other imports to fix imblearn compatibility
try:
    import sklearn.utils._tags as sklearn_tags
except ImportError:
    sklearn_tags = types.ModuleType("sklearn.utils._tags")
    sys.modules["sklearn.utils._tags"] = sklearn_tags

if not hasattr(sklearn_tags, "_safe_tags"):
    def _safe_tags(estimator, key=None):
        tags = {}
        if hasattr(estimator, "_get_tags"):
            try:
                tags = estimator._get_tags()
            except Exception:
                pass
        elif hasattr(estimator, "__sklearn_tags__"):
            try:
                res = estimator.__sklearn_tags__
                if callable(res):
                    tags = res()
                else:
                    tags = res
            except Exception:
                pass
        
        default_tags = {
            "non_categorical": True,
            "requires_y": True,
            "poor_score": False,
            "no_validation": False,
            "multioutput": False,
            "allow_nan": True,
            "stateless": False,
            "multilabel": False,
            "_xfail_checks": False,
        }
        
        merged_tags = {**default_tags, **tags}
        if key is not None:
            return merged_tags.get(key, None)
        return merged_tags

    sklearn_tags._safe_tags = _safe_tags

# Patch AdaBoostClassifier.__init__ to ignore 'algorithm' parameter for imblearn compatibility
from sklearn.ensemble import AdaBoostClassifier
original_init = AdaBoostClassifier.__init__

def patched_init(self, *args, **kwargs):
    kwargs.pop("algorithm", None)
    return original_init(self, *args, **kwargs)

AdaBoostClassifier.__init__ = patched_init

# Patch _RemainderColsList in sklearn.compose._column_transformer for backward compatibility
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList:
        pass
    _ct._RemainderColsList = _RemainderColsList

# ──────────────────────────────────────────────────────────────────
# Standard library imports
# ──────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64
from datetime import datetime, date

# ──────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# Custom CSS — Slate / Teal Premium Theme
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #0f172a;
}

.stApp {
    background-color: #f8fafc;
}

/* ── Hide sidebar, footer, deploy button ── */
section[data-testid="stSidebar"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Banner ── */
.banner-wrapper {
    width: 100%;
    max-height: 220px;
    overflow: hidden;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05);
}
.banner-wrapper img {
    width: 100%;
    height: 220px;
    object-fit: cover;
    display: block;
}

/* ── Header text ── */
.main-header {
    text-align: center;
    padding: 0rem 1rem 1.5rem 1rem;
}
.main-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 0.4rem;
    background: linear-gradient(135deg, #1e293b 0%, #0f766e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header p {
    color: #64748b;
    font-size: 1.05rem;
    margin-top: 0;
}

/* ── Premium Card Containers ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.02) !important;
    margin-bottom: 1.5rem !important;
}

.group-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f766e;
    margin-top: 0;
    margin-bottom: 1.2rem;
    border-bottom: 2px solid #f1f5f9;
    padding-bottom: 0.5rem;
}

/* ── Result cards ── */
.result-card {
    text-align: center;
    border-radius: 16px;
    padding: 2.5rem 1.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
    transition: all 0.3s ease;
}
.result-cancel {
    background: linear-gradient(135deg, #fef2f2 0%, #fff5f5 100%);
    border: 1.5px solid #fecaca;
}
.result-notcancel {
    background: linear-gradient(135deg, #f0fdf4 0%, #f4fbf7 100%);
    border: 1.5px solid #bbf7d0;
}
.result-card .emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-card .label {
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0.5rem 0;
    letter-spacing: 0.05em;
}
.result-cancel .label { color: #b91c1c; }
.result-notcancel .label { color: #047857; }
.result-card .prob {
    font-size: 1.1rem;
    font-weight: 600;
    color: #475569;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0d9488, #115e59) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.2) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(13, 148, 136, 0.3) !important;
}
.stButton > button:active {
    transform: translateY(1px) !important;
}

/* ── Input widgets spacing ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stDateInput > div > div > input {
    border-radius: 10px !important;
    border-color: #cbd5e1 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    color: #334155 !important;
    background-color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Categorical options matching the model
# ──────────────────────────────────────────────────────────────────
HOTEL_OPTIONS = ['Resort Hotel', 'City Hotel']
MONTH_OPTIONS = ['July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March', 'April', 'May', 'June']
MEAL_OPTIONS = ['BB', 'FB', 'HB', 'SC']
MARKET_SEGMENT_OPTIONS = ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups', 'Aviation']
DISTRIBUTION_CHANNEL_OPTIONS = ['Direct', 'Corporate', 'TA/TO', 'GDS']
ROOM_TYPE_OPTIONS = ['A', 'C', 'D', 'E', 'G', 'F', 'H', 'L', 'B']
DEPOSIT_TYPE_OPTIONS = ['No Deposit', 'Refundable', 'Non Refund']
CUSTOMER_TYPE_OPTIONS = ['Transient', 'Contract', 'Transient-Party', 'Group']
DAY_TYPE_OPTIONS = ['Weekday', 'Weekend']

COUNTRY_OPTIONS = [
    'GBR', 'PRT', 'USA', 'ESP', 'IRL', 'FRA', 'ROU', 'NOR', 'OMN', 'ARG', 'POL', 'DEU',
    'BEL', 'CHE', 'CN', 'GRC', 'ITA', 'NLD', 'DNK', 'RUS', 'SWE', 'AUS', 'EST', 'CZE',
    'BRA', 'FIN', 'MOZ', 'BWA', 'LUX', 'SVN', 'ALB', 'IND', 'CHN', 'MEX', 'MAR', 'UKR',
    'SMR', 'LVA', 'PRI', 'SRB', 'CHL', 'AUT', 'BLR', 'LTU', 'TUR', 'ZAF', 'AGO', 'ISR',
    'CYM', 'ZMB', 'CPV', 'ZWE', 'DZA', 'KOR', 'CRI', 'HUN', 'ARE', 'TUN', 'JAM', 'HRV',
    'HKG', 'IRN', 'GEO', 'AND', 'GIB', 'URY', 'JEY', 'CAF', 'CYP', 'COL', 'GGY', 'KWT',
    'NGA', 'MDV', 'VEN', 'SVK', 'FJI', 'KAZ', 'PAK', 'IDN', 'LBN', 'PHL', 'SEN', 'SYC',
    'AZE', 'BHR', 'NZL', 'THA', 'DOM', 'MKD', 'MYS', 'ARM', 'JPN', 'LKA', 'CUB', 'CMR',
    'BIH', 'MUS', 'COM', 'SUR', 'UGA', 'BGR', 'CIV', 'JOR', 'SYR', 'SGP', 'BDI', 'SAU',
    'VNM', 'PLW', 'EGY', 'PER', 'MLT', 'MWI', 'ECU', 'MDG', 'ISL', 'UZB', 'NPL', 'BHS',
    'MAC', 'TGO', 'TWN', 'DJI', 'STP', 'KNA', 'ETH', 'IRQ', 'HND', 'RWA', 'QAT', 'KHM',
    'MCO', 'BGD', 'IMN', 'TJK', 'NIC', 'BEN', 'VGB', 'TZA', 'GAB', 'GHA', 'TMP', 'GLP',
    'KEN', 'LIE', 'GNB', 'MNE', 'UMI', 'MYT', 'FRO', 'MMR', 'PAN', 'BFA', 'LBY', 'MLI',
    'NAM', 'BOL', 'PRY', 'BRB', 'ABW', 'AIA', 'SLV', 'DMA', 'PYF', 'GUY', 'LCA', 'ATA',
    'GTM', 'ASM', 'MRT', 'NCL', 'KIR', 'SDN', 'ATF', 'SLE', 'LAO'
]

# ──────────────────────────────────────────────────────────────────
# Load model (Verifies candidates)
# ──────────────────────────────────────────────────────────────────
candidates = [
    "bestmodel_bookinghotel_2406526_1153.pkl",
    "bestmodel_bookinghotel_240626_1153.pkl"
]
MODEL_PATH = None
for cand in candidates:
    p = os.path.join(os.path.dirname(__file__), cand)
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    # Default fallback
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "bestmodel_bookinghotel_2406526_1153.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ──────────────────────────────────────────────────────────────────
# Header with banner image
# ──────────────────────────────────────────────────────────────────
BANNER_PATH = os.path.join(os.path.dirname(__file__), "hotel_header.png")

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    banner_b64 = get_base64_image(BANNER_PATH)
    st.markdown(f"""
    <div class="banner-wrapper">
        <img src="data:image/png;base64,{banner_b64}" alt="Hotel Banner">
    </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    pass

st.markdown("""
<div class="main-header">
    <h1>🏨 Hotel Booking Cancellation Predictor</h1>
    <p>Predict hotel reservation cancellations instantly using our XGBoost Pipeline Model</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Input Form Grid layout
# ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="medium")

with col_left:
    # ── Group 1: Detail Reservasi & Kedatangan (Arrival & Reservation) ──
    with st.container(border=True):
        st.markdown('<div class="group-title">📅 Detail Reservasi & Kedatangan</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        hotel = c1.selectbox("Tipe Hotel (Hotel Type)", HOTEL_OPTIONS)
        arrival_date = c2.date_input("Tanggal Kedatangan (Arrival Date)", value=date.today())
        
        c3, c4 = st.columns(2)
        lead_time = c3.number_input("Waktu Tunggu (Lead Time - days)", min_value=0, max_value=800, value=15, step=1)
        
        # Calculate default day_type automatically
        # Will be updated by stays duration inputs
        day_type_default = "Weekday"
        day_type = c4.selectbox("Tipe Menginap (Stay Type)", DAY_TYPE_OPTIONS, index=0)

    # ── Group 2: Durasi Menginap & Akomodasi (Duration & Room) ──
    with st.container(border=True):
        st.markdown('<div class="group-title">🌙 Durasi Menginap & Akomodasi</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        weekend_nights = c5.number_input("Malam Akhir Pekan (Weekend Nights)", min_value=0, max_value=20, value=0, step=1)
        week_nights = c6.number_input("Malam Hari Kerja (Week Nights)", min_value=0, max_value=50, value=2, step=1)
        
        c7, c8 = st.columns(2)
        reserved_room_type = c7.selectbox("Tipe Kamar Dipesan (Reserved Room Type)", ROOM_TYPE_OPTIONS)
        deposit_type = c8.selectbox("Tipe Deposit (Deposit Type)", DEPOSIT_TYPE_OPTIONS)

    # ── Group 3: Rincian Tamu & Hidangan (Guests & Meals) ──
    with st.container(border=True):
        st.markdown('<div class="group-title">👥 Rincian Tamu & Hidangan</div>', unsafe_allow_html=True)
        c9, c10, c11 = st.columns(3)
        adults = c9.number_input("Dewasa (Adults)", min_value=1, max_value=10, value=2, step=1)
        children = c10.number_input("Anak-anak (Children)", min_value=0, max_value=10, value=0, step=1)
        babies = c11.number_input("Bayi (Babies)", min_value=0, max_value=10, value=0, step=1)
        
        c12, c13 = st.columns(2)
        meal = c12.selectbox("Paket Hidangan (Meal Plan)", MEAL_OPTIONS)
        country = c13.selectbox("Negara Asal (Guest Country)", COUNTRY_OPTIONS)

with col_right:
    # ── Group 4: Profil & Riwayat Pelanggan (Customer Profile & History) ──
    with st.container(border=True):
        st.markdown('<div class="group-title">📋 Profil & Riwayat Pelanggan</div>', unsafe_allow_html=True)
        c14, c15 = st.columns(2)
        customer_type = c14.selectbox("Tipe Pelanggan (Customer Type)", CUSTOMER_TYPE_OPTIONS)
        is_repeated_guest_val = c15.selectbox("Pelanggan Berulang? (Repeated Guest)", ["No", "Yes"])
        is_repeated_guest = 1 if is_repeated_guest_val == "Yes" else 0
        
        c16, c17 = st.columns(2)
        previous_cancellations = c16.number_input("Pembatalan Sebelumnya (Previous Cancellations)", min_value=0, max_value=30, value=0, step=1)
        previous_bookings_not_canceled = c17.number_input("Pemesanan Sukses Sebelumnya (Previous Non-Cancellations)", min_value=0, max_value=80, value=0, step=1)

    # ── Group 5: Saluran Distribusi & Pembuat Reservasi (Distribution & Agents) ──
    with st.container(border=True):
        st.markdown('<div class="group-title">📡 Saluran Distribusi & Pembuat Reservasi</div>', unsafe_allow_html=True)
        c18, c19 = st.columns(2)
        market_segment = c18.selectbox("Segmen Pasar (Market Segment)", MARKET_SEGMENT_OPTIONS)
        distribution_channel = c19.selectbox("Saluran Distribusi (Distribution Channel)", DISTRIBUTION_CHANNEL_OPTIONS)
        
        c20, c21 = st.columns(2)
        is_agent_val = c20.selectbox("Melalui Agen? (Booked via Agent)", ["No", "Yes"])
        is_company_val = c21.selectbox("Melalui Perusahaan? (Booked via Company)", ["No", "Yes"])
        is_agent = 1 if is_agent_val == "Yes" else 0
        is_company = 1 if is_company_val == "Yes" else 0

    # ── Group 6: Keuangan & Permintaan Khusus (Financials & Requests) ──
    with st.container(border=True):
        st.markdown('<div class="group-title">💰 Keuangan & Permintaan Khusus</div>', unsafe_allow_html=True)
        c22, c23 = st.columns(2)
        adr = c22.number_input("Tarif Harian Rata-rata (ADR)", min_value=0.0, max_value=5000.0, value=85.0, step=1.0, format="%.2f")
        booking_changes = c23.number_input("Perubahan Pesanan (Booking Changes)", min_value=0, max_value=30, value=0, step=1)
        
        c24, c25, c26 = st.columns(3)
        days_in_waiting_list = c24.number_input("Hari di Waiting List", min_value=0, max_value=500, value=0, step=1)
        required_car_parking_spaces = c25.number_input("Parkir Mobil", min_value=0, max_value=10, value=0, step=1)
        total_of_special_requests = c26.number_input("Permintaan Khusus", min_value=0, max_value=10, value=0, step=1)

# ──────────────────────────────────────────────────────────────────
# Calculations & Features Processing
# ──────────────────────────────────────────────────────────────────
# Extract date features
arrival_date_year = arrival_date.year
arrival_date_month = arrival_date.strftime('%B')
arrival_date_week_number = int(arrival_date.isocalendar()[1])
arrival_date_day_of_month = arrival_date.day

# Stay calculations
stay_duration = int(weekend_nights + week_nights)

# Guest calculations
total_tamu = int(adults + children + babies)
is_family = 1 if (children > 0 or babies > 0) else 0

# Predict button
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
if st.button("🔮 Mulai Prediksi Pembatalan (Predict Cancellation)"):
    # Build input dataframe matching columns exactly (order and types)
    input_dict = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_date_year,
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": arrival_date_week_number,
        "arrival_date_day_of_month": arrival_date_day_of_month,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "booking_changes": booking_changes,
        "deposit_type": deposit_type,
        "days_in_waiting_list": days_in_waiting_list,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "total_tamu": total_tamu,
        "stay_duration": stay_duration,
        "is_family": is_family,
        "day_type": day_type,
        "is_agent": is_agent,
        "is_company": is_company,
    }

    input_df = pd.DataFrame([input_dict])

    try:
        model = load_model()
        prediction = model.predict(input_df)[0]

        # Get probabilities
        try:
            # TunedThresholdClassifierCV handles prediction and probabilities internally
            proba = model.predict_proba(input_df)[0]
            prob_cancel = proba[1]
            prob_not = proba[0]
        except Exception:
            prob_cancel = None
            prob_not = None

        res_left, res_mid, res_right = st.columns([1, 2, 1])
        with res_mid:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card result-cancel">
                    <div class="emoji">❌</div>
                    <div class="label">CANCELLED</div>
                    <div class="prob">{"Probability: {:.0%}".format(prob_cancel) if prob_cancel is not None else ""}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-notcancel">
                    <div class="emoji">✅</div>
                    <div class="label">NOT CANCELLED</div>
                    <div class="prob"<{"Probability: {:.0%}".format(prob_not) if prob_not is not None else ""}</div>
                </div>
                """, unsafe_allow_html=True)

        # Show input summary in expander
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        with st.expander("📄 Rincian Data Input (Input Data Summary)"):
            st.dataframe(input_df.T.rename(columns={0: "Nilai (Value)"}), use_container_width=True)

    except FileNotFoundError:
        st.error(f"❌ File Model tidak ditemukan di: `{MODEL_PATH}`. Silakan letakkan file .pkl model di folder yang sama.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan prediksi: {e}")
