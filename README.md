# Twitter Sentiment Analysis with LSTM & GRU

A deep learning project that classifies Twitter sentiments as **positive** or **negative** using LSTM and GRU models. Dive into the world of NLP and uncover the vibes of tweets! 😄

## 🚀 Features
- Cleans and preprocesses tweets (removes URLs, mentions, hashtags, stopwords)
- Trains **LSTM** and **GRU** models to predict sentiment
- Evaluates models with accuracy, confusion matrices, and classification reports
- Predicts sentiment on new tweets with a simple function
- Visualizes training performance with accuracy/loss plots 📊

## 📂 Project Structure
- `Twitter_Sentiment_Analysis_LSTM_GRU.ipynb`: Jupyter notebook with the full pipeline (preprocessing, training, evaluation, prediction)
- `requirements.txt`: Dependency list for reproducibility
- `README.md`: You're reading it! 😎

## 🧠 Models
- **LSTM Model**: Recurrent neural network for sequence modeling, ~79.3% test accuracy
- **GRU Model**: Lightweight alternative to LSTM, ~79.2% test accuracy
- Built with TensorFlow/Keras and trained on the Sentiment140 dataset

👉 **Dataset**: Download the [Sentiment140 dataset](https://www.kaggle.com/kazanova/sentiment140) from Kaggle and place `training.1600000.processed.noemoticon.csv` in the project root or update the path in the notebook.

## 📦 Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
