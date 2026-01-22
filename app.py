import streamlit as st
import pandas as pd
import joblib
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sleep Condition Predictor",
    page_icon="😴",
    layout="centered"
)

st.title("😴 Sleep Condition Prediction System")
st.write(
    "This system predicts sleep condition (**Refreshed / Tired**) "
    "using a **Linear Discriminant Analysis (LDA)** model based on "
    "sleep behaviour and lifestyle factors."
)

# =====================================================
# LOAD DATASET (FOR DROPDOWN OPTIONS ONLY)
# =====================================================
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSfZ5jkjpEVm2yq-fr6GXdzLnC7FBc068Yv0uknFBdSov90M47Fzgu8SZDxdZwPyRZbyB_eIlLJ8us4/pub?output=csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df_data = load_data()

# =====================================================
# LOAD MODEL (DEPLOY SAFE)
# =====================================================
MODEL_PATH = "sleep_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file (sleep_model.pkl) not found.")
    st.stop()

model = joblib.load(MODEL_PATH)
st.sidebar.success("✅ LDA Model Loaded")

# Numeric features used in LDA
FEATURES = [
    "age",
    "caffeineintake",
    "exerciseduration",
    "sleeptime",
    "screentime",
    "awakeningn",
    "noise"
]

# =====================================================
# QUESTIONNAIRE (19 QUESTIONS)
# =====================================================
st.header("📝 Sleep Quality Questionnaire")

# -------------------------------
# Section 1: Demographic
# -------------------------------
st.subheader("1️⃣ Personal Information")

age = st.slider("Age", 15, 80, 22)

gender = st.selectbox(
    "Gender",
    sorted(df_data["gender"].dropna().unique())
)

occupation = st.selectbox(
    "Occupation",
    sorted(df_data["occupation"].dropna().unique())
)

# -------------------------------
# Section 2: Lifestyle Habits
# -------------------------------
st.subheader("2️⃣ Lifestyle Habits")

consumecaffeine = st.radio(
    "Do you consume caffeine daily?",
    ["Yes", "No"]
)

caffeineintake = st.slider(
    "Caffeine intake (cups per day)",
    0, 10, 1
)

smoke = st.radio(
    "Do you smoke?",
    ["Yes", "No"]
)

exercise = st.radio(
    "Do you exercise regularly?",
    ["Yes", "No"]
)

exerciseduration = st.slider(
    "Exercise duration per day (minutes)",
    0, 300, 30
)

# -------------------------------
# Section 3: Sleep Behaviour
# -------------------------------
st.subheader("3️⃣ Sleep Behaviour")

phonebeforesleep = st.selectbox(
    "How often do you use your phone before sleep?",
    sorted(df_data["phonebeforesleep"].dropna().unique())
)

screentime = st.slider(
    "Screen time before sleep (hours)",
    0, 12, 3
)

middaynap = st.selectbox(
    "Do you take midday naps?",
    sorted(df_data["middaynap"].dropna().unique())
)

awakeningn = st.slider(
    "Number of times you wake up at night",
    0, 10, 1
)

# -------------------------------
# Section 4: Environment & Mental State
# -------------------------------
st.subheader("4️⃣ Sleep Environment & Mental State")

sharedroom = st.radio(
    "Do you share your room with others?",
    ["Yes", "No"]
)

anxiety = st.selectbox(
    "How often do you feel anxious?",
    sorted(df_data["anxiety"].dropna().unique())
)

roombright = st.selectbox(
    "Room brightness during sleep",
    sorted(df_data["roombright"].dropna().unique())
)

noise = st.slider(
    "Noise level during sleep (1 = very quiet, 5 = very noisy)",
    1, 5, 2
)

bedtime = st.time_input("Usual bedtime")
uptime = st.time_input("Usual wake-up time")

sleeptime = st.slider(
    "Average sleep duration per night (hours)",
    0, 12, 6
)

# =====================================================
# PREDICTION
# =====================================================
st.divider()

if st.button("🔮 Predict Sleep Condition"):
    # Prepare numeric input for LDA
    input_df = pd.DataFrame(
        [[
            age,
            caffeineintake,
            exerciseduration,
            sleeptime,
            screentime,
            awakeningn,
            noise
        ]],
        columns=FEATURES
    )

    prediction = model.predict(input_df)[0]

    st.subheader("🧠 Prediction Result")

    if prediction.lower() == "refreshed":
        st.success("✅ You are likely to wake up feeling **REFRESHED**.")
    else:
        st.warning("⚠️ You are likely to wake up feeling **TIRED**.")

    # =================================================
    # SIMPLE RULE-BASED ADVICE
    # =================================================
    st.subheader("💡 Personalised Suggestions")

    if phonebeforesleep == "Always":
        st.info("📵 Reduce phone usage before bedtime to improve sleep quality.")

    if caffeineintake >= 4:
        st.info("☕ High caffeine intake may negatively affect sleep quality.")

    if sleeptime < 6:
        st.info("⏰ Try to increase sleep duration to at least 7 hours.")

    if anxiety != "Never":
        st.info("🧘 Relaxation techniques may help reduce anxiety before sleep.")
