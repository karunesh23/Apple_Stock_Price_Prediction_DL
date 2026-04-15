# 📈 Apple Stock Price Prediction

A Deep Learning project that predicts Apple stock prices using historical data. The model is trained on past stock prices and uses neural networks to forecast future trends.

---

## 🚀 Features
- Time series forecasting using Deep Learning  
- Data preprocessing & scaling using MinMaxScaler  
- Model training with Neural Networks (Keras/TensorFlow)  
- Performance evaluation using R² Score  
- Visualization of loss and predictions  
- Model & scaler saving for future use  

---

## 📂 Project Structure
```
Apple_Stock_Price_Prediction/
│── .gitignore
│── APPLE.csv
│── app.py
│── train.py
│── predict.py
│── model.h5
│── scaler.pkl
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository
```
git clone https://github.com/karunesh23/Apple_Stock_Price_Prediction.git
cd Apple_Stock_Price_Prediction
```

### 2️⃣ Create Virtual Environment
```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

---

## 🧠 Model Training
```
python train.py
```

👉 This will:
- Train the deep learning model  
- Calculate R² score  
- Save model as `model.h5`  
- Save scaler as `scaler.pkl`  

---

## 🔮 Prediction
```
python predict.py
```

---

## 🖥️ Run Streamlit App
```
streamlit run app.py
```

---

## 📊 Dataset
- File: `APPLE.csv`  
- Contains historical Apple stock data  
- Features include Date, Open, High, Low, Close, Volume  

---

## 📈 Model Details
- Model Type: Deep Neural Network  
- Layers: Dense (128 → 64 → 32 → 1)  
- Activation: ReLU  
- Loss Function: Mean Squared Error  
- Optimizer: Adam  
- Evaluation Metric: R² Score  

---

## 📊 Visualizations
- Training vs Validation Loss Graph  
- Actual vs Predicted Stock Prices  

---

## 📌 Future Improvements
- Use LSTM for better time series prediction  
- Hyperparameter tuning  
- Add real-time stock API  
- Deploy on cloud (AWS/Render)  

---

## ⚠️ Important Notes
- Do not upload `venv/` folder  
- Ensure `requirements.txt` is updated  
- Model performance depends on data quality  

---

## 👨‍💻 Author

**Karunesh Bansal**

📧 **Email:** karuneshbansal53@gmail.com  

💼 **LinkedIn:**  
[https://www.linkedin.com/in/karunesh-bansal](https://www.linkedin.com/in/karunesh-bansal)

---

## ⭐ Support
If you like this project, give it a ⭐ on GitHub!
