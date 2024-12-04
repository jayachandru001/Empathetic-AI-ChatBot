import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from transformers import pipeline
import google.generativeai as genai
from openai import OpenAI
import os

# Constants for the number of classes
EMO_NUM_CLASSES = 6
SEN_NUM_CLASSES = 3

# Load the tokenizer and base model for emotion classification and sentiment classification
base_model_path = 'meta-llama/Llama-3.2-1B'
loaded_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Emotion classification model
loaded_base_model_emotion = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=EMO_NUM_CLASSES)
adapter_path1 = "jayachandru001/Llama-3.2-1B-for-Emotion-clf-v1"
emotionClf = PeftModel.from_pretrained(loaded_base_model_emotion, adapter_path1)

# Sentiment classification model
loaded_base_model_sentiment = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=SEN_NUM_CLASSES)
adapter_path2 = "jayachandru001/sentiment_classification_using_llama_3.2_1B_V2"
sentimentClf = PeftModel.from_pretrained(loaded_base_model_sentiment, adapter_path2)

# Load the Hugging Face pipelines
sentiment_clf = pipeline("text-classification", sentimentClf, tokenizer=loaded_tokenizer)
emotion_clf = pipeline("text-classification", emotionClf, tokenizer=loaded_tokenizer)

# Configure the generative AI model with the API key
secret_value_0 = os.getenv('GOOGLE_API_KEY')
secret_value_1 = os.getenv('HF_TOKEN')
secret_value_3 = os.getenv("SAMBANOVA_API_KEY")



genai.configure(api_key=secret_value_0)

# Initialize the generative model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 256,
    "response_mime_type": "text/plain",
}
sys_instructions = """You are an advanced AI designed to generate empathetic 
                      responses based on the emotional state of the user, which has been classified by 
                      a separate emotional classification model. Your task is to generate a supportive, 
                      contextually appropriate response that aligns with the identified emotion, offering 
                      comfort, validation, and empathy."""

# Initialize Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-flash", 
                                     generation_config=generation_config,
                                     system_instruction=sys_instructions)

# Generate the empathetic response from Sambanova's Meta-Llama (modified to match Gemini's flow)
opeaiClient = OpenAI(api_key= secret_value_3, 
                base_url="https://api.sambanova.ai/v1",)


# Emotion labels
emotion_id2label = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

# Sentiment labels
sentiment_id2label = {
    0: "Neutral",
    1: "Positive",
    2: "Negative",
}

# Function to handle the chatbot's conversation logic
def chat_with_user(user_message, history, model_choice):
    # Get sentiment prediction
    sentiment_result = sentiment_clf(user_message)
    pred_sentiment = int(sentiment_result[0]['label'].split('_')[1])
    sentiment = sentiment_id2label[pred_sentiment]

    emotion = ""
    # If the sentiment is not neutral, predict emotion
    if pred_sentiment != 0:
        # Get emotion prediction
        pred_label = emotion_clf(user_message)
        pred_emotion = int(pred_label[0]['label'].split('_')[1])
        # Map the prediction to the emotion label
        emotion = emotion_id2label[pred_emotion]

    # Prepare the user's message with detected sentiment and emotion, each on a new line
    user_input_emotion = f"{user_message}\nSentiment: [{sentiment}]\nEmotion: [{emotion}]"
    
    # Respond based on model selection
    if model_choice == "Gemini AI":
        # Generate the empathetic response from Gemini (same as original method)
        chat = gemini_model.start_chat(history=[])
        response = chat.send_message(user_input_emotion)
        temp_response = ""
        for chunk in response:
            temp_response += chunk.text
    elif model_choice == "Meta-Llama":
        llama_sys_instructions = [{"role": "system", "content": sys_instructions},
                      {"role": "user", "content": user_input_emotion}]
        response = opeaiClient.chat.completions.create(model="Meta-Llama-3.1-8B-Instruct",
                                             messages= llama_sys_instructions,
                                             temperature=0.5,
                                             top_p=0.5
                                            )
        
        temp_response = response.choices[0].message.content
    
    # Return sentiment, emotion, and the AI's empathetic response
    history.append((
        f"User: {user_message}\nSentiment: {sentiment}\nEmotion: {emotion}",
        f"Bot: {temp_response}"
    ))
    return "", history

# Define the interface components with Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Chat with Empathetic AI Bot")

    # Dropdown for selecting response generator
    model_dropdown = gr.Dropdown(["Meta-Llama", "Gemini AI"], label="Select Response Generator")

    # Chatbot container (left = user, right = bot)
    chat = gr.Chatbot()

    # Textbox for user input at the bottom of the screen
    textbox = gr.Textbox(placeholder="Type a message...", show_label=False)

    # Submit button and action to handle message
    textbox.submit(chat_with_user, inputs=[textbox, chat, model_dropdown], outputs=[textbox, chat])

    # Inject custom CSS directly via gr.HTML
    gr.HTML("""
    <style>
        .chatbot {
            max-height: 80vh;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
        }
        .chatbot .message:nth-child(odd) { 
            text-align: left; 
            background-color: #e1f5fe; 
            border-radius: 5px;
            padding: 10px;
            margin: 5px;
        }
        .chatbot .message:nth-child(even) { 
            text-align: right; 
            background-color: #f1f8e9; 
            border-radius: 5px;
            padding: 10px;
            margin: 5px;
        }
        .gradio-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .gradio-input {
            position: fixed;
            bottom: 20px;
            left: 10%;
            right: 10%;
            width: 80%;
        }
    </style>
    """)

    # Inject custom JavaScript to automatically scroll to the bottom when a new message is added
    gr.HTML("""
    <script>
        const chatbot = document.querySelector('.chatbot');
        const inputBox = document.querySelector('.gradio-input input');
        // Scroll to the bottom of the chat after each update
        function scrollToBottom() {
            chatbot.scrollTop = chatbot.scrollHeight;
        }
        // Listen for any new message updates
        const observer = new MutationObserver(scrollToBottom);
        observer.observe(chatbot, { childList: true });
        // Initial scroll position when the page loads
        window.onload = scrollToBottom;
        // Ensure that input field is at the bottom of the page
        inputBox.addEventListener('focus', () => {
            setTimeout(scrollToBottom, 100);
        });
    </script>
    """)

# Launch the Gradio app
demo.launch()