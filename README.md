Twitter Sentiment Analysis with LSTM & GRU
A deep learning project that classifies Twitter sentiments as positive or negative using LSTM and GRU models. Dive into the world of NLP and uncover the vibes of tweets! 😄

🚀 Features
Cleans and preprocesses tweets (removes URLs, mentions, hashtags, stopwords)
Trains LSTM and GRU models to predict sentiment
Evaluates models with accuracy, confusion matrices, and classification reports
Predicts sentiment on new tweets with a simple function
Visualizes training performance with accuracy/loss plots 📊
📂 Project Structure
Twitter_Sentiment_Analysis_LSTM_GRU.ipynb: Jupyter notebook with the full pipeline (preprocessing, training, evaluation, prediction)
requirements.txt: Dependency list for reproducibility
README.md: You're reading it! 😎
🧠 Models
LSTM Model: Recurrent neural network for sequence modeling, ~79.3% test accuracy
GRU Model: Lightweight alternative to LSTM, ~79.2% test accuracy
Built with TensorFlow/Keras and trained on the Sentiment140 dataset
👉 Dataset: Download the Sentiment140 dataset from Kaggle and place training.1600000.processed.noemoticon.csv in the project root or update the path in the notebook.

📦 Requirements
Install dependencies using:

bash

Copy
pip install -r requirements.txt
Key Dependencies:

Python 3.10
TensorFlow 2.17.1
Pandas 2.2.2
NLTK 3.9.1
scikit-learn 1.6.0
NumPy 1.26.4
Matplotlib 3.9.2
Seaborn 0.13.2
See requirements.txt for the full list.

🛠️ Setup Instructions
Clone the Repository:
bash

Copy
git clone https://github.com/your-username/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
Install Dependencies:
bash

Copy
pip install -r requirements.txt
Download the Dataset:
Grab the Sentiment140 dataset from Kaggle.
Place training.1600000.processed.noemoticon.csv in the project directory or update the pd.read_csv() path in the notebook.
Run the Notebook:
Open Twitter_Sentiment_Analysis_LSTM_GRU.ipynb in Jupyter Notebook or Google Colab.
Run all cells to preprocess data, train models, evaluate, and predict sentiments.
💡 Pro Tip: Use Google Colab with GPU runtime for faster training!
🎯 Usage
Predict sentiment on new tweets using the predict_sentiment function in the notebook. Example:

python

Copy
new_tweet = "I love this product, it is amazing!"
lstm_pred = predict_sentiment(new_tweet, model_lstm, tokenizer, max_len)
gru_pred = predict_sentiment(new_tweet, model_gru, tokenizer, max_len)
print(f'LSTM Prediction: {lstm_pred}')  # Output: Positive
print(f'GRU Prediction: {gru_pred}')  # Output: Positive
📈 Results
LSTM: ~79.3% test accuracy after 5 epochs
GRU: ~79.2% test accuracy after 5 epochs
Visuals include:
Confusion matrices for both models
Training accuracy and loss plots 📉
Check the notebook for detailed classification reports!
📜 License
This project is licensed under the MIT License - see the  file for details.

🙌 Acknowledgments
Sentiment140 dataset by Kaggle
Powered by TensorFlow and NLTK
Built with ❤️ for NLP enthusiasts!
