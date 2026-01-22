import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sleep Condition Predictor",
    page_icon="😴",
    layout="centered"
)

st.title("😴 Sleep Condition Prediction System")
st.write(
    "This application predicts sleep condition (Refreshed / Tired) "
    "using a Linear Discriminant Analysis (LDA) model."
)

# =========================
# LOAD MODEL (DEPLOY SAFE)
# =========================
MODEL_PATH = "sleep_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file (sleep_model.pkl) not found.")
    st.stop()

# Load ONLY the LDA model
model = joblib.load(MODEL_PATH)

# Features used during training
FEATURES = [
    "age",
    "caffeineintake",
    "exerciseduration",
    "sleeptime",
    "screentime",
    "awakeningn",
    "noise"
]

st.sidebar.success("✅ LDA Model Loaded Successfully")

# =========================
# USER INPUT (19 QUESTIONS UI)
# =========================
st.header("📝 Sleep Questionnaire")

# --- Section 1: Demographic ---
st.subheader("1️⃣ Personal Information")
age = st.slider("Age", 15, 80, 22)

# --- Section 2: Lifestyle ---
st.subheader("2️⃣ Lifestyle Habits")

consumecaffeine = st.radio(
    "Do you consume caffeine daily?",
    ["Yes", "No"]
)

caffeineintake = st.slider(
    "Caffeine intake (cups per day)",
    0, 10, 1
)

exercise = st.radio(
    "Do you exercise regularly?",
    ["Yes", "No"]
)

exerciseduration = st.slider(
    "Exercise duration per day (minutes)",
    0, 300, 30
)

smoke = st.radio(
    "Do you smoke?",
    ["Yes", "No"]
)

# --- Section 3: Sleep Behaviour ---
st.subheader("3️⃣ Sleep Behaviour")

phonebeforesleep = st.selectbox(
    "How often do you use your phone before sleep?",
    ["Never", "Sometimes", "Always"]
)

screentime = st.slider(
    "Screen time before sleep (hours)",
    0, 12, 3
)

middaynap = st.selectbox(
    "Do you take midday naps?",
    ["Never", "Occasionally", "Frequently"]
)

awakeningn = st.slider(
    "Number of times you wake up at night",
    0, 10, 1
)

# --- Section 4: Environment & Mental State ---
st.subheader("4️⃣ Sleep Environment & Mental State")

anxiety = st.selectbox(
    "How often do you feel anxious?",
    ["Never", "Sometimes", "Always"]
)

roombright = st.selectbox(
    "Room brightness during sleep",
    ["Completely dark", "Dim light", "Bright"]
)

noise = st.slider(
    "Noise level during sleep (1 = very quiet, 5 = very noisy)",
    1, 5, 2
)

sleeptime = st.slider(
    "Average sleep duration per night (hours)",
    0, 12, 6
)

sharedroom = st.radio(
    "Do you share your room with others?",
    ["Yes", "No"]
)

bedtime = st.time_input("Usual bedtime")
uptime = st.time_input("Usual wake-up time")

occupation = st.text_input("Occupation")

# =========================
# PREDICTION
# =========================
st.divider()

if st.button("🔮 Predict Sleep Condition"):
    # Prepare input for model (numeric features only)
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

    # LDA prediction
    prediction = model.predict(input_df.values)[0]

    st.subheader("🧠 Prediction Result")

    if prediction.lower() == "refreshed":
        st.success("✅ You are likely to wake up feeling **REFRESHED**.")
    else:
        st.warning("⚠️ You are likely to wake up feeling **TIRED**.")

    # =========================
    # SIMPLE ADVICE (NON-ML)
    # =========================
    st.subheader("💡 Personalised Suggestions")

    if phonebeforesleep == "Always":
        st.info("📵 Reduce phone usage before bedtime.")

    if caffeineintake >= 4:
        st.info("☕ High caffeine intake may affect sleep quality.")

    if sleeptime < 6:
        st.info("⏰ Try to increase your sleep duration to at least 7 hours.")

    if anxiety != "Never":
        st.info("🧘 Relaxation techniques may help reduce anxiety before sleep.")
