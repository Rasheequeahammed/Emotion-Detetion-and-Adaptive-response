# Emotional Sentiment Analysis and Adaptive Response System 🤖💬
An AI-based chatbot designed for culturally sensitive emotional support. This project integrates emotional sentiment analysis and empathetic response generation using Natural Language Processing (NLP) techniques.

---

## **Industry** 💻
Technology

## **Department** 🧠
AI/ML / NLP

## **Product/Process** 🛠️
AI-based Chatbot for Culturally Sensitive Emotional Support

---

## **Project Description** 📖
The Emotional Sentiment Analysis and Adaptive Response System is an AI-driven chatbot designed to identify users' emotional states through conversational input. By leveraging the DistilBERT model and Natural Language Processing techniques, the chatbot provides culturally sensitive and empathetic responses tailored to user emotions. This system is aimed at offering mental health support.

---

## **Key Features** 🔑
- Sentiment analysis of user input using the GoEmotions dataset 🧑‍💻.
- Contextual and culturally sensitive response generation 🌍.
- Interactive chatbot interface built using Streamlit 📱.
- Real-time emotion prediction and adaptive response generation 🎯.

---

## **Key Objectives** 🎯
1. Develop a machine learning model to identify emotional states from text data 🧠.
2. Design empathetic and contextually appropriate response generation systems 💬.
3. Ensure responses are culturally sensitive and tailored to users 🌏.
4. Integrate sentiment analysis and response generation models into a cohesive AI chatbot prototype 🤖.
5. Continuously evaluate and improve performance through user feedback 🔄.

---

## **Project Steps** 🛠️

### 1. **Data Collection** 📊
- Dataset: [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions/tree/6d5e11c91321f16b1909cd0042a7770af3aca55a) (via Hugging Face).

### 2. **Data Preprocessing** 🧹
- Lowercasing text 🔠.
- Removing URLs, HTML tags, numbers, and punctuation 🧹.
- Removing stopwords using NLTK 🚫.
- Lemmatization for generalization 🌱.
- Tokenization using Hugging Face's tokenizer 💻.
- Padding and truncation for uniform input lengths 📏.

### 3. **Data Balancing** ⚖️
- Addressed class imbalance using RandomUnderSampler from the `imblearn` library ⚖️.

### 4. **Model Development** 🏗️
- Model: **DistilBERT** (a smaller, efficient version of BERT).
- Hyperparameter Tuning:
  - Adjusted learning rate, batch size, epochs, and dropout rate 🔧.
  - Achieved 67.77% accuracy after fine-tuning (up from 54%) 📈.

### 5. **Model Deployment** 🚀
- Deployed using **Streamlit**.
- Integrated emotion prediction with empathetic response generation 💬.

---

## **Challenges Faced** ⚠️
1. **Resource Limitations**: Frequent GPU usage limits in Google Colab 🚧.
2. **Data Imbalance**: Addressed using undersampling techniques ⚖️.
3. **Training Time**: Overcame slow model training with optimized hyperparameters 🕰️.

---

## **Deployment** 🌐
The project is deployed as a real-time chatbot using **Streamlit**. Users can interact with the bot through a user-friendly interface, with responses styled dynamically based on emotional sentiment.

---

## **Technologies Used** 🛠️
- **Languages**: Python 🐍
- **Libraries**:
  - NLP: Hugging Face Transformers, NLTK 🧠
  - Data Preprocessing: Pandas, NumPy, Scikit-learn 📊
  - Data Balancing: `imblearn` ⚖️
  - Deployment: Streamlit 📱
- **Models**: DistilBERT 🤖
- **Dataset**: [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions/tree/6d5e11c91321f16b1909cd0042a7770af3aca55a) 📚

---

## **Installation** ⚙️
1. Clone this repository:
   ```bash
   git clone https://github.com/Rasheequeahammed/your-repo-name.git
