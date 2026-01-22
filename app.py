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

st.title("😴 Sleep Quality Questionnaire & Predictor")
st.write("Please answer all questions below. Your sleep condition will be predicted based on your responses.")

# =========================
# LOAD MODEL
# =========================
if os.path.exists("sleep_model.pkl"):
    saved = joblib.load("sleep_model.pkl")
    model = saved["model"]
    imputer = saved["imputer"]
    features = saved["features"]

    st.sidebar.success("✅ Model Loaded")

    # =========================
    # SECTION 1: PERSONAL INFORMATION
    # =========================
    st.subheader("👤 Section 1: Personal Information")

    gender = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 15, 80, 23)
    occupation = st.selectbox(
        "Occupation",
        ["Student", "Working Adult", "Self-Employed", "Unemployed"]
    )

    # =========================
    # SECTION 2: LIFESTYLE HABITS
    # =========================
    st.subheader("☕ Section 2: Lifestyle Habits")

    consumecaffeine = st.radio("Do you consume caffeine daily?", ["Yes", "No"])
    caffeineintake = st.slider("Caffeine intake (cups per day)", 0, 10, 1)
    smoke = st.radio("Do you smoke?", ["Yes", "No"])
    exercise = st.radio("Do you exercise regularly?", ["Yes", "No"])
    exerciseduration = st.slider("Exercise duration per day (minutes)", 0, 300, 20)

    # =========================
    # SECTION 3: SLEEP BEHAVIOUR
    # =========================
    st.subheader("📱 Section 3: Sleep Behaviour")

    phonebeforesleep = st.selectbox(
        "How often do you use your phone before sleep?",
        ["Never", "Sometimes", "Always"]
    )

    screentime = st.slider("Screen time before sleep (hours)", 0, 12, 4)

    middaynap = st.selectbox(
        "Do you take midday naps?",
        ["Never", "Occasionally", "Frequently"]
    )

    awakeningn = st.slider("Number of times you wake up at night", 0, 10, 0)

    # =========================
    # SECTION 4: ENVIRONMENT & MENTAL STATE
    # =========================
    st.subheader("🛏️ Section 4: Sleep Environment & Mental State")

    sharedroom = st.radio("Do you share your sleeping room?", ["Yes", "No"])

    anxiety = st.radio(
        "How often do you feel anxious?",
        ["Never", "Sometimes", "Always"]
    )

    roombright = st.selectbox(
        "Room brightness during sleep",
        ["Completely dark", "Dim light", "Bright"]
    )

    bedtime = st.time_input("Usual bedtime")
    uptime = st.time_input("Usual wake-up time")

    sleeptime = st.slider("Average sleep duration per night (hours)", 0, 12, 6)

    noise = st.slider(
        "Noise level during sleep (1 = very quiet, 5 = very noisy)",
        1, 5, 2
    )

    # =========================
    # PREDICTION
    # =========================
    if st.button("🔮 Predict My Sleep Condition"):
        # Only numeric features go to ML model
        df = pd.DataFrame([[age, caffeineintake, exerciseduration,
                            sleeptime, screentime, awakeningn, noise]],
                          columns=features)

        df_clean = imputer.transform(df)
        prediction = model.predict(df_clean)[0]

        st.divider()
        st.header(f"🧠 Prediction Result: {prediction.upper()}")

        # =========================
        # FEEDBACK
        # =========================
        if prediction.lower() == "refreshed":
            st.success("✅ You are likely to wake up feeling refreshed.")
        else:
            st.warning("⚠️ You may feel tired. Consider improving your sleep habits.")

        st.subheader("💡 Personalised Advice")

        if phonebeforesleep == "Always":
            st.info("📵 Reduce phone usage before sleep to improve sleep quality.")

        if anxiety != "Never":
            st.info("🧘 Try relaxation techniques to reduce anxiety before bedtime.")

        if screentime > 5:
            st.info("⏰ Reduce screen exposure at least 1 hour before sleep.")

else:
    st.error("❌ sleep_model.pkl NOT FOUND")
