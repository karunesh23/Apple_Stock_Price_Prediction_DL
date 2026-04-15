import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predict import predict_price

st.set_page_config(page_title="Apple Price Prediction", layout="centered")

st.title("📈 Apple Stock Price Prediction (ANN)")
st.write("Upload your dataset to predict next closing price")

uploaded_file = st.file_uploader("Upload AAPL.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    st.subheader("📊 Data Preview")
    st.write(df.tail())

    # Debug
    st.subheader("🔍 Last 5 Close Values")
    st.write(df['Close'].tail())

    # Chart
    st.subheader("📉 Closing Price Trend")
    fig = plt.figure()
    plt.plot(df['Date'], df['Close'])
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    if len(df) >= 60:
        last_60 = df['Close'].values[-60:]

        if st.button("🚀 Predict Next Price"):
            result = predict_price(last_60)

            st.success(f"💰 Predicted Next Closing Price: ${result:.2f}")

            last_actual = df['Close'].values[-1]

            st.write(f"📊 Last Actual Price: ${last_actual:.2f}")

            diff = abs(result - last_actual)

            if diff < 5:
                st.success("✅ Prediction looks good")
            elif diff < 20:
                st.warning("⚠️ Prediction slightly off")
            else:
                st.error("❌ Prediction seems wrong (check model/data)")
    else:
        st.warning("Dataset must have at least 60 rows!")