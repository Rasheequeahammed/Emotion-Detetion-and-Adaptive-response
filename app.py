import streamlit as st
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import base64

# Load the saved model and tokenizer
model_save_path = "C:\\Users\\rasheeque raheem\\Documents\\MODERATE LTD\\BLACK box code\\DOCUMENTATION\\saved_model\\"
model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)

# Set the model to evaluation mode
model.eval()

# Define the device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to predict the emotion from a given text
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    label_mapping = {
        0: 'anger', 1: 'approval', 2: 'fear', 3: 'disgust', 4: 'admiration',
        5: 'love', 6: 'disapproval', 7: 'surprise', 8: 'joy', 9: 'excitement',
        10: 'curiosity', 11: 'realization', 12: 'amusement', 13: 'sadness',
        14: 'annoyance', 15: 'embarrassment', 16: 'gratitude', 17: 'confusion',
        18: 'caring', 19: 'desire', 20: 'optimism', 21: 'remorse', 22: 'disappointment',
        23: 'nervousness', 24: 'grief', 25: 'relief', 26: 'pride'
    }
    return label_mapping.get(predicted_class, "Unknown Emotion")

# Function to generate an empathetic response based on the emotion
def generate_empathetic_response(emotion):
    response = ""
    if emotion == 'anger':
        response = "😠 I understand you're upset. Let's take a deep breath together. It's okay to feel angry, but we can work through it."
    elif emotion == 'approval':
        response = "👍 It's great to see you're feeling positive and approving of something. Keep spreading the good vibes!"
    elif emotion == 'fear':
        response = "😟 It's completely okay to feel anxious. You're not alone in this. Take things one step at a time, and you'll get through it."
    elif emotion == 'disgust':
        response = "🤢 It's tough to deal with unpleasant feelings. Sometimes it's best to distance yourself from what causes them."
    elif emotion == 'admiration':
        response = "👏 Admiration is a powerful feeling. It's wonderful that you have such respect and appreciation."
    elif emotion == 'love':
        response = "❤️ Love is such a beautiful feeling. It's amazing to see you so positive and affectionate!"
    elif emotion == 'disapproval':
        response = "😒 I can sense your disapproval. It's okay to disagree with things; expressing yourself is important."
    elif emotion == 'surprise':
        response = "😲 Wow, that's surprising! I can see how unexpected things can bring such an emotional response."
    elif emotion == 'joy':
        response = "😊 That's wonderful! I'm so happy to hear you're feeling great. Keep spreading that positivity!"
    elif emotion == 'excitement':
        response = "🎉 Your excitement is contagious! I'm thrilled to see your enthusiasm—it's such a positive energy!"
    elif emotion == 'curiosity':
        response = "🤔 Curiosity is such a powerful tool! Keep exploring and asking questions; it's how we learn and grow."
    elif emotion == 'realization':
        response = "💡 Realizations can be life-changing. It's wonderful that you're discovering new insights about yourself or the world."
    elif emotion == 'amusement':
        response = "😂 Laughter is the best medicine! It's amazing how humor can uplift our spirits."
    elif emotion == 'sadness':
        response = "😔 I'm really sorry you're feeling this way. I'm here for you. It's okay to feel sad sometimes."
    elif emotion == 'annoyance':
        response = "😤 I know things can be frustrating. It's okay to feel annoyed; just take a moment for yourself."
    elif emotion == 'embarrassment':
        response = "😳 Embarrassment happens to all of us. Don't be too hard on yourself; we all go through it."
    elif emotion == 'gratitude':
        response = "🙏 Gratitude brings so much peace. It's beautiful that you're appreciating the good things around you."
    elif emotion == 'remorse':
        response = "😞 Remorse can be difficult to carry, but it's a sign that you're self-aware. Take time to forgive yourself."
    elif emotion == 'disappointment':
        response = "😔 It's tough when things don't turn out the way we hoped. Take a deep breath, and know that better days are ahead."
    elif emotion == 'caring':
        response = "🤗 Caring for others is such a noble feeling. Your compassion makes the world a better place."
    elif emotion == 'desire':
        response = "🔥 Desire is a powerful motivator. It's great that you have the drive to go after what you want!"
    elif emotion == 'optimism':
        response = "🌟 Optimism is a beautiful trait. Keep believing in the best of things—it makes the world brighter!"
    elif emotion == 'remorse':
        response = "😔 I understand you feel remorseful. It's a sign of your moral compass. Take it as a step toward growth."
    elif emotion == 'nervousness':
        response = "😬 Nervousness is normal when you're facing something important. Take deep breaths; you've got this!"
    elif emotion == 'grief':
        response = "💔 I'm deeply sorry for your grief. It's okay to mourn and take your time to heal. You're not alone."
    elif emotion == 'relief':
        response = "😌 I'm glad you're feeling a sense of relief. It must feel good to have that weight lifted off your shoulders."
    elif emotion == 'pride':
        response = "🎖️ Pride in your achievements is well-earned! Celebrate your successes—you deserve it."
    else:
        response = "💬 It's okay to feel what you're feeling. Whatever it is, I'm here to listen."

    return response
# Function to set the background of the app

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        .chat-container {{
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: black;  /* Set response font color to black */
            font-weight: bold;  /* Make response font bold */
        }}
        .input-box {{
            background: rgba(0, 0, 0, 0.7);
            color: white;
            backdrop-filter: blur(8px);
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }}
        .header {{
            position: absolute;
            top: 10px;
            left: 10px;
            font-size: 24px;
            font-weight: bold;
            color: white;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            padding: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Apply the background image
set_background(r"C:\Users\rasheeque raheem\Documents\MODERATE LTD\BLACK box code\DOCUMENTATION\BGIMG.jpg")

# Add your name with a hyperlink at the top-left corner
st.markdown(
    """
    <style>
    .header {
        position: fixed;
        top: 60px;
        left: 10px;
        font-size: 16px;
        color: white;
        background-color: rgba(0, 0, 0, 0.7);
        padding: 8px 12px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        z-index: 1000;
    }
    .header a {
        text-decoration: none;
        color: #00acee; /* LinkedIn blue */
        font-weight: bold;
    }
    </style>
    <div class="header">
        Done by <a href="https://www.linkedin.com/in/rasheeque-ahammed-raheem-aa777325a" target="_blank">Rasheeque 👨‍💻</a>
    </div>
    """,
    unsafe_allow_html=True
)


# Emotion-to-color mapping
emotion_color_mapping = {
    'anger': {'font': 'red', 'background': 'rgba(255, 255, 200, 1.0)'},  # Red font, light background
    'joy': {'font': 'yellow', 'background': 'rgba(50, 50, 50, 0.9)'},  # Yellow font, dark background
    'sadness': {'font': 'blue', 'background': 'rgba(200, 200, 255, 0.8)'},  # Blue font, light background
    'surprise': {'font': 'orange', 'background': 'rgba(0, 0,0, 0.8)'},  # Orange font, light background
    'fear': {'font': 'purple', 'background': 'rgba(240, 200, 240, 0.8)'},  # Purple font, light background
    'love': {'font': 'pink', 'background': 'rgba(0, 0, 128, 0.8)'},  # Pink font, light background
    'disgust': {'font': 'green', 'background': 'rgba(255, 255, 0, 0.8)'},  # Green font, light background
    # Default styling for undefined emotions
    'default': {'font': 'black', 'background': 'rgba(255, 255, 255, 0.8)'}  
}

# Generate the CSS for each response dynamically based on emotion
def generate_response_style(emotion):
    color_style = emotion_color_mapping.get(emotion, emotion_color_mapping['default'])
    font_color = color_style['font']
    background_color = color_style['background']
    return f"color: {font_color}; background: {background_color}; padding: 15px; border-radius: 10px; font-weight: bold;"


# Main content
st.title('Emotional Sentiment Analysis & Adaptive Response System')
st.write("This app simulates a chat with emotion-based empathetic responses.")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to display chat
def display_chat():
    for msg in reversed(st.session_state.messages):  # Show latest input first
        if msg['role'] == 'user':
            st.markdown(
                f"<div class='input-box'>{msg['text']}</div>",
                unsafe_allow_html=True
            )
        else:
            emotion = msg.get('emotion', 'default')  # Get emotion for the response
            style = generate_response_style(emotion)
            st.markdown(
                f"<div style='{style}'>{msg['text']}</div>",
                unsafe_allow_html=True
            )
            
            
# Input box for user messages

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

def clear_input():
    st.session_state.input_text = ""  # Reset the input text
user_input = st.text_input("Type your message:", st.session_state.input_text, key="input_box", on_change=clear_input)


if user_input:
    st.session_state.messages.append({'role': 'user', 'text': user_input})
    emotion = predict(user_input)
    response = generate_empathetic_response(emotion)
    st.session_state.messages.append({'role': 'bot', 'text': response, 'emotion': emotion})
    display_chat()
else:
    display_chat()