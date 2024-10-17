import threading
import traceback
import json
import threading
import re
import nltk
import spacy
import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import time
from openai import OpenAI
import base64
import streamlit.components.v1 as components
import plotly.graph_objects as go
from pathlib import Path
import os
from typing import Dict, List, Union
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge
from bert_score import score
from collections import defaultdict
import logging
from streamlit_shadcn_ui import button, card
from textblob import TextBlob
import transformers
from dotenv import load_dotenv
st.set_page_config(page_title="Smart Greenhouse Monitor", layout="wide", initial_sidebar_state="collapsed")

load_dotenv()

def setup_openai_client():
    # Get API key from environment
    os.environ["openai_secret_key"] = st.secrets["openai_secret_key"]

    # Ensure the API key is set correctly
    if "openai_secret_key" not in os.environ:
        st.error("OpenAI API key is not set. Please check your secrets configuration.")
    else:
        import openai
        openai.api_key = os.environ["openai_secret_key"]
        return openai

    
transformers.logging.set_verbosity_error()
nltk.download('punkt_tab')
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
# Image paths
image_path = "venlo.webp"
gif_path = "basil.gif"

def project_explanation_page():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
    }
    .logo {
        width: 100px;
    }
    .gif-container {
        width: 150px;
        height: 150px;
        overflow: hidden;
        border-radius: 50%;
        margin: 0 20px;
    }
    .title {
        font-size: 2.5em;
        color: #2c3e50;
        text-align: center;
        margin: 20px 0;
    }
    .content {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with logos and GIF
    st.markdown("""
    <div class="header-container">
        <img src="data:image/svg+xml;base64,{}" class="logo">
        <div class="gif-container">
            <img src="data:image/gif;base64,{}" style="width: 100%; height: 100%; object-fit: cover;">
        </div>
        <img src="data:image/svg+xml;base64,{}" class="logo">
    </div>
    """.format(
        get_base64_of_bin_file("TUC_LogoText_TUC.svg"),
        get_base64_of_bin_file("basil.gif"),
        get_base64_of_bin_file("ACSD_Logo_TUC.svg")
    ), unsafe_allow_html=True)

    st.markdown('<h1 class="title">Welcome to Greenhouse Chatbot / Willkommen beim Gewächshaus-Chatbot</h1>', unsafe_allow_html=True)

    # Language selection
    language = st.radio("Select language / Sprache auswählen", ("English", "Deutsch"))

    st.markdown('<div class="content">', unsafe_allow_html=True)

    if language == "English":
        st.write("""
        Our Smart Greenhouse Chatbot is designed to help you analyze and understand greenhouse data with ease. Whether you're a farmer, researcher, or greenhouse manager, this tool will assist you in interpreting complex environmental data and optimizing your greenhouse operations.

        ## How to Use the Chatbot

        1. **Date Selection**: Start by providing a date for which you want to analyze greenhouse data.
        2. **Feature Selection**: Choose which environmental features you'd like to visualize.
        3. **Data Visualization**: View plots of the selected features.
        4. **Ask Questions**: Inquire about the data and receive insights.
        5. **Feedback**: Provide feedback to improve the chatbot's performance.

        ## Example Questions

        - "What was the temperature trend throughout the day?"
        - "How did CO2 levels correlate with plant growth?"
        - "Were there any sudden changes in humidity levels?"
        - "What recommendations do you have for improving energy efficiency based on today's data?"

        Feel free to start by selecting a date and exploring your greenhouse data!
        """)

    else:
        st.write("""
        Unser Smart-Gewächshaus-Chatbot wurde entwickelt, um Ihnen die Analyse und das Verständnis von Gewächshausdaten zu erleichtern. Ob Sie Landwirt, Forscher oder Gewächshausmanager sind, dieses Tool unterstützt Sie bei der Interpretation komplexer Umweltdaten und der Optimierung Ihres Gewächshausbetriebs.

        ## Wie man den Chatbot benutzt

        1. **Datumsauswahl**: Beginnen Sie mit der Angabe eines Datums, für das Sie Gewächshausdaten analysieren möchten.
        2. **Merkmalsauswahl**: Wählen Sie, welche Umweltmerkmale Sie visualisieren möchten.
        3. **Datenvisualisierung**: Betrachten Sie Diagramme der ausgewählten Merkmale.
        4. **Fragen stellen**: Erkundigen Sie sich über die Daten und erhalten Sie Erkenntnisse.
        5. **Feedback**: Geben Sie Feedback, um die Leistung des Chatbots zu verbessern.

        ## Beispielfragen

        - "Wie war der Temperaturverlauf über den Tag?"
        - "Wie korrelierten die CO2-Werte mit dem Pflanzenwachstum?"
        - "Gab es plötzliche Änderungen in der Luftfeuchtigkeit?"
        - "Welche Empfehlungen haben Sie zur Verbesserung der Energieeffizienz basierend auf den heutigen Daten?"

        Fangen Sie an, indem Sie ein Datum auswählen und Ihre Gewächshausdaten erkunden!
        """)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Proceed to NDA / Weiter zur Vertraulichkeitsvereinbarung"):
        st.session_state.page = "nda"
        st.rerun()


def welcome_page():
    gif_path = Path("basil.gif")
    image_path = Path("venlo.webp")
    # CSS for styling
    st.markdown("""
    <style>
    .container {
        max-width: 900px;
        margin: 0 auto;
        text-align: center;
    }
    .title {
        font-size: 2.5em;
        color: #2c3e50;
        margin-bottom: 20px;
        animation: fadeIn 2s;
    }
    .image-container {
        position: relative;
        width: 100%;
        height: 450px;
        background-color: transparent;
        margin-bottom: 20px;
        border-radius: 8px;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .small-loader {
        width: 60px;
        margin-bottom: 20px;
    }
    .big-loader {
        width: 120px;
        margin-bottom: 20px;
    }
    .description {
        color: #34495e;
        line-height: 1.6;
        animation: slideIn 2s;
    }
    .leaf {
        display: inline-block;
        animation: rotate 3s infinite linear;
    }
    .button-container {
        text-align: center;
        margin-top: 30px;
    }
    .stButton button {
        display: block !important;
        margin: 0 auto !important;
        padding: 10px 20px !important;
        font-size: 1.2em !important;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

    # Show the loading GIF first (big loader for the initial page load)
    # Show the loading GIF first
    gif_data = get_base64_of_bin_file(gif_path)
    if gif_data:
        st.markdown(f"""
        <div class="container">
            <img class="big-loader" src="data:image/gif;base64,{gif_data}" alt="Loading...">
        </div>
        """, unsafe_allow_html=True)
        time.sleep(5)  # Simulating delay of 5 seconds
    else:
        st.warning("Loading animation could not be displayed.")

    # After loading, display the main content
    image_data = get_base64_of_bin_file(image_path)
    if image_data:
        st.markdown(f"""
        <div class="container">
            <h1 class="title">Welcome to the Greenhouse Chatbot</h1>
            <div class="image-container" id="imageContainer">
                <img id="greenhouseImage" src="data:image/webp;base64,{image_data}" alt="Greenhouse" style="display: block; width: 90%; height: auto;">
            </div>
            <p class="description">
                This chatbot allows you to interact with various aspects of your greenhouse environment. 
                <span class="leaf">🍃</span> Grow with us! <span class="leaf">🍅</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Main image could not be loaded. Please check the file path.")

    # Functional center button using Streamlit
    if st.button("Enter Greenhouse", key="enter_greenhouse_center"):
        if gif_data:
            st.markdown(f"""
            <div class="container">
                <img class="small-loader" src="data:image/gif;base64,{gif_data}" alt="Loading...">
            </div>
            """, unsafe_allow_html=True)

        time.sleep(2)  # Simulate a small delay before the next page loads
        st.session_state.page = "monitor"
        st.rerun()


def welcome_form():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .title {
        font-size: 2.5em;
        color: #2c3e50;
        text-align: center;
        margin: 20px 0;
    }

    .gif-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .form-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Greenhouse GIF
    st.markdown("""
    <div class="gif-container">
        <img src="data:image/gif;base64,{}" style="width: 300px; height: 300px; border-radius: 10px;">
    </div>
    """.format(get_base64_of_bin_file("basil.gif")), unsafe_allow_html=True)
    # Title
    st.markdown('<h1 class="title">Welcome to the Smart Greenhouse Chatbot</h1>', unsafe_allow_html=True)
    # Form
    with st.form("welcome_form"):
        name = st.text_input("Name")
        profession = st.selectbox("Profession", ["", "Farmer", "Student", "Researcher", "Specialist", "Other"])
        specialty = ""
        other_profession = ""
        if profession == "Specialist":
            specialty = st.selectbox("Specialty", ["", "Agriculture", "Horticulture", "Plant Science", "Microbiology"])
        elif profession == "Other":
            other_profession = st.text_input("Please specify your profession")

        submitted = st.form_submit_button("Submit")

        if submitted:
            if profession == "Other":
                profession = other_profession
            elif profession == "Specialist":
                profession = f"Specialist - {specialty}"

            if not name or not profession or (profession.startswith("Specialist") and not specialty):
                st.error("Please fill in all fields.")
            else:
                user_info = {
                    "name": name,
                    "profession": profession,
                    "timestamp": datetime.now().isoformat()
                }
                user_entry = save_user_data(user_info=user_info)
                st.success("User information saved successfully!")
                st.session_state.user_info = user_entry
                st.session_state.page = "welcome"
                st.rerun()

    if 'user_info' in st.session_state:
        st.session_state.page = "welcome"
        st.rerun()


def nda_page():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
    }
    .logo {
        width: 100px;
    }
    .title {
        font-size: 2.5em;
        color: #2c3e50;
        text-align: center;
        margin: 20px 0;
    }
    .content {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .language-selector {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .gif-container {
        width: 150px;
        height: 150px;
        overflow: hidden;
        border-radius: 50%;
        margin: 0 20px;
    }
    .agreement-section {
        margin-top: 20px;
        padding: 15px;
        background-color: #e9ecef;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with logos
    st.markdown("""
    <div class="header-container">
        <img src="data:image/svg+xml;base64,{}" class="logo">
        <div class="gif-container">
            <img src="data:image/gif;base64,{}" style="width: 100%; height: 100%; object-fit: cover;">
        </div>
        <img src="data:image/svg+xml;base64,{}" class="logo">
    </div>
    """.format(
        get_base64_of_bin_file("TUC_LogoText_TUC.svg"),
        get_base64_of_bin_file("basil.gif"),
        get_base64_of_bin_file("ACSD_Logo_TUC.svg")
    ), unsafe_allow_html=True)

    st.markdown('<h1 class="title">Non-Disclosure Agreement / Vertraulichkeitsvereinbarung</h1>', unsafe_allow_html=True)

    # Language selection
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    language = st.radio("Select language / Sprache auswählen", ("English", "Deutsch"))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content">', unsafe_allow_html=True)

    if language == "English":
        st.title("Non-Disclosure Agreement")
        st.write("""
        This interaction is recorded for research purposes by the ACSD team, 
        which is part of the Electrical Engineering and Information Technology (ETIT) department at TU Chemnitz.

        The information collected will be used to help advance the chatbot and improve its functionality.

        By proceeding, you agree to the following terms and conditions:

        1. During this interaction, we may collect certain personal information, such as your name and professional background. However, this personally identifiable information (PII) will not be used in the research. Only the content of your prompts and responses will be analyzed and utilized for research purposes.
        2. Your participation is voluntary, and you may discontinue at any time.
        3. The data will be securely stored and will not be shared with third parties without your explicit consent.
        4. The ACSD team will retain the collected data for a period necessary to achieve the research objectives. Upon completion of the research or at your request, data may be securely deleted in accordance with applicable data protection laws.
        5. The chatbot's responses are for research purposes only and should not be construed as legal, medical, or professional advice. The ACSD team makes no warranties regarding the accuracy or completeness of the information provided by the chatbot during the research phase.
        6. During this interaction, you will only be prompted or asked questions related to the specific project topics, including model predictive control (MPC), horticulture, plant growth, or other directly related subjects. You will not be asked questions outside of these domains, and the collected data will not be used for any other purpose than advancing research within these fields.
        7. This NDA complies with applicable data protection and privacy laws, including but not limited to the GDPR, CCPA, or other relevant local laws.
        By proceeding, you agree to these terms and conditions.
        """)
    else:
        st.title("Vertraulichkeitsvereinbarung (Non-Disclosure Agreement)")
        st.write("""
        Diese Interaktion wird zu Forschungszwecken vom ACSD-Team aufgezeichnet, 
        das Teil des Fachbereichs Elektro- und Informationstechnik (ETIT) der TU Chemnitz ist.

        Die gesammelten Informationen werden verwendet, um den Chatbot weiterzuentwickeln und seine Funktionalität zu verbessern.

        Mit dem Fortfahren stimmen Sie den folgenden Bedingungen zu:

        1. Während dieser Interaktion können wir bestimmte persönliche Informationen sammeln, wie z.B. Ihren Namen und beruflichen Hintergrund. Diese personenbezogenen Daten (PII) werden jedoch nicht in der Forschung verwendet. Nur der Inhalt Ihrer Eingaben und Antworten wird analysiert und für Forschungszwecke verwendet.
        2. Ihre Teilnahme ist freiwillig, und Sie können jederzeit abbrechen.
        3. Die Daten werden sicher gespeichert und ohne Ihre ausdrückliche Zustimmung nicht an Dritte weitergegeben.
        4. Das ACSD-Team wird die gesammelten Daten für einen Zeitraum aufbewahren, der zur Erreichung der Forschungsziele notwendig ist. Nach Abschluss der Forschung oder auf Ihre Anfrage hin können die Daten gemäß den geltenden Datenschutzgesetzen sicher gelöscht werden.
        5. Die Antworten des Chatbots dienen nur zu Forschungszwecken und sollten nicht als rechtliche, medizinische oder berufliche Beratung verstanden werden. Das ACSD-Team übernimmt keine Gewähr für die Richtigkeit oder Vollständigkeit der während der Forschungsphase vom Chatbot bereitgestellten Informationen.
        6. Während dieser Interaktion werden Sie nur zu spezifischen Projektthemen befragt, einschließlich modellbasierter prädiktiver Regelung (MPC), Gartenbau, Pflanzenwachstum oder anderen direkt verwandten Themen. Sie werden nicht zu Themen außerhalb dieser Bereiche befragt, und die gesammelten Daten werden nicht für andere Zwecke als die Förderung der Forschung in diesen Bereichen verwendet.
        7. Diese Vertraulichkeitsvereinbarung entspricht den geltenden Datenschutz- und Privatsphäre-Gesetzen, einschließlich, aber nicht beschränkt auf die DSGVO, CCPA oder andere relevante lokale Gesetze.
        Mit dem Fortfahren stimmen Sie diesen Bedingungen zu.
        """)

    # Radio button for agreement
    agree = st.radio("Do you agree to the terms of the Non-Disclosure Agreement?", ("Yes", "No"), index=0)

    # Submission logic
    if st.button("Submit"):
        if agree == "Yes":
            save_user_data(nda_agreed=True)
            st.session_state.nda_agreed = True
            st.session_state.page = "welcome_form"
            st.rerun()
        else:  # Only other option is "No"
            save_user_data(nda_agreed=False)
            st.session_state.nda_agreed = False
            st.write("Thank you for your interest. For any inquiries, please contact:")
            st.write("ramesh.naagarajan@etit.tu-chemnitz.de")
            st.write("Prof Stefan Streif - stefan.streif@etit.tu-chemnitz.de")

def save_user_data(user_info=None, nda_agreed=None, chat_messages=None):
    # Load existing data
    try:
        with open("user_data.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    user_entry = None

    if user_info:
        # Check if user already exists
        user_entry = next((item for item in data if
                           item.get("name") == user_info["name"] and
                           item.get("profession") == user_info["profession"]), None)

        if not user_entry:
            # Create new entry if user doesn't exist
            user_entry = {
                "name": user_info["name"],
                "profession": user_info["profession"],
                "timestamp": datetime.now().isoformat(),
                "nda_agreed": True,  # Set NDA agreement to true by default
                "chat_history": []
            }
            data.append(user_entry)
    else:
        # If updating existing user
        if 'user_info' in st.session_state:
            user_entry = next((item for item in data if
                               item.get("name") == st.session_state.user_info.get("name") and
                               item.get("profession") == st.session_state.user_info.get("profession")), None)

    # Update NDA agreement only if explicitly set to False
    if nda_agreed is False:
        if user_entry:
            user_entry["nda_agreed"] = False
            st.session_state.nda_agreed = False
        else:
            # Create a temporary entry for NDA agreement
            temp_entry = next((item for item in data if "temp_nda" in item), None)
            if temp_entry is None:
                temp_entry = {"temp_nda": True, "nda_agreed": True}  # Set to True by default
                data.append(temp_entry)
            temp_entry["nda_agreed"] = False  # Only update if explicitly set to False
            st.session_state.nda_agreed = False

    # Add chat messages
    if chat_messages and user_entry:
        if "chat_history" not in user_entry:
            user_entry["chat_history"] = []

        # Save only the text content of the messages
        serializable_messages = []
        for message in chat_messages:
            serializable_message = {
                "role": message["role"],
                "content": message["content"]
            }
            serializable_messages.append(serializable_message)

        user_entry["chat_history"].append({
            "messages": serializable_messages,
            "timestamp": datetime.now().isoformat()
        })

    # Save updated data
    with open("user_data.json", "w") as f:
        json.dump(data, f, indent=2)

    return user_entry


def apply_custom_css():
    # Get base64 encoded images for the chatbot icons and send button
    leaves_gif = get_base64_of_bin_file('leaves.gif')
    sprout_gif = get_base64_of_bin_file('sprout.gif')
    leaff_gif = get_base64_of_bin_file('leaff.gif')

    custom_css = f"""
    <style>
        /* Style for user messages */
        .stChatMessage div[data-testid="StChatMessageAvatar"] {{
            background-image: url('data:image/gif;base64,{leaves_gif}') !important;
            background-size: cover !important;
        }}
        
        /* Style for assistant messages */
        .stChatMessage div[data-testid="StChatMessageAvatar"].assistant {{
            background-image: url('data:image/gif;base64,{sprout_gif}') !important;
            background-size: cover !important;
        }}
        
        /* Style for send button */
        .stChatInputContainer button.stSendButton {{
            background-image: url('data:image/gif;base64,{leaff_gif}') !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            background-color: transparent !important;
            border: none !important;
            width: 40px !important;
            height: 40px !important;
        }}
        
        .stChatInputContainer button.stSendButton svg {{
            display: none !important;
        }}

        /* Style for input box */
        .stTextInput input {{
            border-radius: 20px;
            border: 2px solid #4CAF50;
            padding: 10px 15px;
        }}

        /* Replace arrow with custom GIF in input box */
        .stTextInput::after {{
            content: '';
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            width: 20px;
            height: 20px;
            background-image: url('data:image/gif;base64,{leaff_gif}');
            background-size: contain;
            background-repeat: no-repeat;
        }}

        /* Style for monitor page */
        .monitor-container {{
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .monitor-heading {{
            color: #1E90FF;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }}

        .monitor-chart {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }}
        
        /* Heading color globally */
        h1, h2, h3, h4, h5, h6 {{
            color: #1E90FF !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    # JavaScript to ensure the button stays visible and apply custom styles
    st.components.v1.html(
        """
        <script>
            document.addEventListener('DOMContentLoaded', (event) => {
                console.log("Custom JS is running");
                const observer = new MutationObserver((mutations) => {
                    mutations.forEach((mutation) => {
                        if (mutation.addedNodes.length) {
                            const sendButton = document.querySelector('button.stSendButton');
                            if (sendButton) {
                                sendButton.style.display = 'block';
                                sendButton.style.opacity = '1';
                            }
            
                            // Apply custom styles to chat avatars
                            const avatars = document.querySelectorAll('.stChatMessage [data-testid="StChatMessageAvatar"]');
                            avatars.forEach((avatar, index) => {
                                if (index % 2 === 0) {
                                    avatar.classList.add('assistant');
                                }
                            });
                        }
                    });
                });
            
                const targetNode = document.body;
                if (targetNode) {
                    observer.observe(targetNode, { childList: true, subtree: true });
                }
            });
        </script>
        """
    )

def install_spacy_model():
    try:
        # Try loading the model to check if it's already installed
        nlp = spacy.load("en-core-web-sm")
    except OSError:
        # If the model is not found, install it
        st.write("Model 'en-core-web-sm' not found. Downloading...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en-core-web-sm"])
        nlp = spacy.load("en-core-web-sm")
    return nlp

class FeedbackSystem:
    def __init__(self, num_chunks):
        self.chunk_weights = np.ones(num_chunks)
        self.feedback_history = []
        self.adjustment_factor = 0.1

    def update_weights(self, question, chunk_embeddings, feedback):
        relevant_indices = find_relevant_chunks(question, chunk_embeddings)
        adjustment = self.adjustment_factor if feedback in ['Yes', 'Partially'] else -self.adjustment_factor
        for idx in relevant_indices:
            self.chunk_weights[idx] = max(0.1, min(2.0, self.chunk_weights[idx] + adjustment))
        self.chunk_weights = np.clip(self.chunk_weights, 0.1, 2.0)
        self.chunk_weights /= np.sum(self.chunk_weights)

    def get_weighted_chunks(self, chunks):
        return [chunk for chunk, weight in sorted(zip(chunks, self.chunk_weights), key=lambda x: x[1], reverse=True)]

    def get_feedback_summary(self):
        total_feedback = len(self.feedback_history)
        helpful_count = sum(1 for _, feedback in self.feedback_history if feedback == 'helpful')
        return {
            'total_feedback': total_feedback,
            'helpful_percentage': (helpful_count / total_feedback) * 100 if total_feedback > 0 else 0,
            'top_chunks': np.argsort(self.chunk_weights)[-5:][::-1].tolist()
        }

class FeedbackEvaluator:
    def __init__(self, json_file_path: str = 'feedback_metrics.json'):
        self.json_file_path = os.path.abspath(json_file_path)
        self.nlp = install_spacy_model
        self._lock = threading.Lock()
        self.metrics = self.load_metrics()
        self.rouge = Rouge()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Initializing FeedbackEvaluator with file path: {self.json_file_path}")

    def evaluate_response(self, response: str, reference: str, question: str) -> Dict[str, float]:
        try:
            # Calculate ROUGE score
            rouge_scores = self.rouge.get_scores(response, reference)[0]
            rouge_l_f1 = rouge_scores['rouge-l']['f']
        except Exception as e:
            logging.error(f"Error calculating ROUGE score: {e}")
            rouge_l_f1 = 0

        try:
            # Calculate similarity for relevance
            question_embedding = self.model.encode([question])
            response_embedding = self.model.encode([response])
            relevance = cosine_similarity(question_embedding, response_embedding)[0][0]
        except Exception as e:
            logging.error(f"Error calculating relevance: {e}")
            relevance = 0

        return {
            'rouge_score': rouge_l_f1,
            'relevance': relevance
        }

    def load_metrics(self) -> Dict:
        default_metrics = {
            "total_feedback": 0,
            "positive_feedback": 0,
            "total_response_quality_score": 0.0,
            "average_response_quality": 0.0,
            "user_satisfaction_rate": 0.0,
            "feedback_history": [],
            "user_metrics": {}
        }

        try:
            if os.path.exists(self.json_file_path):
                with open(self.json_file_path, 'r') as f:
                    loaded_metrics = json.load(f)
                    # Ensure all default keys exist
                    for key, value in default_metrics.items():
                        if key not in loaded_metrics:
                            loaded_metrics[key] = value
                    return loaded_metrics
            return default_metrics
        except Exception as e:
            logging.error(f"Error loading metrics from {self.json_file_path}: {e}")
            return default_metrics

    def save_metrics(self) -> bool:
        try:
            def convert_to_serializable(obj):
                if isinstance(obj, np.float32):
                    return float(obj)
                elif isinstance(obj, defaultdict):
                    return dict(obj)
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            serializable_metrics = json.loads(
                json.dumps(self.metrics, default=convert_to_serializable)
            )

            with open(self.json_file_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)

            print(f"Metrics saved successfully to {self.json_file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving metrics to {self.json_file_path}: {str(e)}")
            print(f"Failed to save metrics: {str(e)}")
            return False

    def add_feedback(self, feedback: str, user_type: str, question: str = None, answer: str = None) -> bool:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            feedback_value = 1 if feedback in ['Yes', 'Partially'] else 0

            # Calculate response quality
            response_quality_score = 0.0
            if question and answer:
                evaluation = self.evaluate_response(answer, answer, question)
                response_quality_score = (evaluation['rouge_score'] + evaluation['relevance']) / 2

            feedback_entry = {
                "timestamp": current_time,
                "feedback": feedback,
                "value": feedback_value,
                "user_type": user_type,
                "question": question,
                "answer": answer,
                "response_quality_score": response_quality_score
            }

            with self._lock:
                # Ensure all required keys exist
                if "total_response_quality_score" not in self.metrics:
                    self.metrics["total_response_quality_score"] = 0.0

                # Update metrics
                self.metrics["total_feedback"] += 1
                self.metrics["total_response_quality_score"] += response_quality_score
                if feedback_value == 1:
                    self.metrics["positive_feedback"] += 1

                self.metrics["feedback_history"].append(feedback_entry)

                # Update user metrics
                if user_type not in self.metrics["user_metrics"]:
                    self.metrics["user_metrics"][user_type] = {"total_feedback": 0, "positive_feedback": 0}
                self.metrics["user_metrics"][user_type]["total_feedback"] += 1
                if feedback_value == 1:
                    self.metrics["user_metrics"][user_type]["positive_feedback"] += 1

                # Update averages
                if self.metrics["total_feedback"] > 0:
                    self.metrics["average_response_quality"] = self.metrics["total_response_quality_score"] / self.metrics["total_feedback"]
                    self.metrics["user_satisfaction_rate"] = (self.metrics["positive_feedback"] / self.metrics["total_feedback"]) * 100

            # Save metrics immediately
            return self.save_metrics()
        except Exception as e:
            logging.error(f"Error adding feedback: {e}")
            print(f"Failed to add feedback: {str(e)}")
            return False


class LLMResponseEvaluator:
    def __init__(self, json_file_path='llm_metrics.json'):
        self.json_file_path = json_file_path
        self.load_metrics()
        self.rouge = Rouge()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_embeddings(self, text):
        return self.model.encode(text)

    def load_metrics(self):
        try:
            with open(self.json_file_path, 'r') as f:
                self.metrics = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.metrics = {
                "total_responses": 0,
                "average_rouge": 0,
                "average_bert_score": 0,
                "average_relevance": 0,
                "human_feedback": defaultdict(int),
                "daily_metrics": defaultdict(lambda: {
                    "total_responses": 0,
                    "average_rouge": 0,
                    "average_bert_score": 0,
                    "average_relevance": 0,
                    "human_feedback": defaultdict(int)
                })
            }

    def evaluate_response(self, response, reference, question, human_feedback=None):

        # Calculate ROUGE score
        try:
            rouge_scores = self.rouge.get_scores(response, reference)[0]
            rouge_l_f1 = rouge_scores['rouge-l']['f']
        except Exception:
            rouge_l_f1 = 0

        # Calculate BERTScore
        try:
            _, _, bert_f1 = score([response], [reference], lang='en', verbose=False)
            bert_score = bert_f1.item()
        except Exception:
            bert_score = 0

        # Calculate relevance
        try:
            question_embedding = self.create_embeddings(question)
            response_embedding = self.create_embeddings(response)
            relevance = cosine_similarity([question_embedding], [response_embedding])[0][0]
        except Exception:
            relevance = 0

        # Update metrics
        self.update_metrics(rouge_l_f1, bert_score, relevance, human_feedback)

        return {
            "rouge_score": rouge_l_f1,
            "bert_score": bert_score,
            "relevance": relevance
        }

    def update_metrics(self, rouge_score, bert_score, relevance, human_feedback):
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Update overall metrics
        self.metrics["total_responses"] += 1
        self.metrics["average_rouge"] = self.running_average(self.metrics["average_rouge"], rouge_score, self.metrics["total_responses"])
        self.metrics["average_bert_score"] = self.running_average(self.metrics["average_bert_score"], bert_score, self.metrics["total_responses"])
        self.metrics["average_relevance"] = self.running_average(self.metrics["average_relevance"], relevance, self.metrics["total_responses"])

        if human_feedback:
            self.metrics["human_feedback"][human_feedback] += 1

        # Update daily metrics
        if current_date not in self.metrics["daily_metrics"]:
            self.metrics["daily_metrics"][current_date] = {
                "total_responses": 0,
                "average_rouge": 0,
                "average_bert_score": 0,
                "average_relevance": 0,
                "human_feedback": defaultdict(int)
            }

        daily_metrics = self.metrics["daily_metrics"][current_date]
        daily_metrics["total_responses"] += 1
        daily_metrics["average_rouge"] = self.running_average(daily_metrics["average_rouge"], rouge_score, daily_metrics["total_responses"])
        daily_metrics["average_bert_score"] = self.running_average(daily_metrics["average_bert_score"], bert_score, daily_metrics["total_responses"])
        daily_metrics["average_relevance"] = self.running_average(daily_metrics["average_relevance"], relevance, daily_metrics["total_responses"])

        if human_feedback:
            daily_metrics["human_feedback"][human_feedback] += 1

        self.save_metrics()

    def running_average(self, current_avg, new_value, n):
        return (current_avg * (n - 1) + new_value) / n

    def save_metrics(self):
        def convert_to_serializable(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, defaultdict):
                return dict(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        serializable_metrics = json.loads(
            json.dumps(self.metrics, default=convert_to_serializable)
        )

        with open(self.json_file_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)

    def get_metrics_summary(self):
        return {
            "total_responses": self.metrics["total_responses"],
            "average_rouge": self.metrics["average_rouge"],
            "average_bert_score": self.metrics["average_bert_score"],
            "average_relevance": self.metrics["average_relevance"],
            "human_feedback": dict(self.metrics["human_feedback"]),
            "daily_metrics": self.metrics["daily_metrics"]
        }


def generate_time_series_plot(df_day, feature_key, target_date, feature_name_mapping):
    try:
        # Convert target_date to datetime if it's a string
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d')
        elif isinstance(target_date, datetime):
            pass
        else:
            raise ValueError(f"Unexpected type for target_date: {type(target_date)}")

        # Create time series
        time_series = pd.date_range(start=target_date.replace(hour=0, minute=0, second=0),
                                    end=target_date.replace(hour=23, minute=55, second=0),
                                    freq='5T')

        # Get feature name and unit
        feature_name, feature_unit = feature_name_mapping[feature_key]

        # Create the plot
        fig = go.Figure()

        # Add the data trace
        fig.add_trace(go.Scatter(
            x=time_series,
            y=df_day[feature_key],
            mode='lines',
            name=feature_name,
            hovertemplate='Time: %{x|%H:%M}<br>' +
                          f'{feature_name}: ' +
                          '%{y:.2f} ' + feature_unit +
                          '<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title=f"{feature_name} on {target_date.strftime('%B %d, %Y')}",
            xaxis_title="Time",
            yaxis_title=f"{feature_name} ({feature_unit})",
            hovermode="x unified",
            xaxis=dict(
                tickformat="%H:%M",
                tickmode="array",
                tickvals=pd.date_range(start=target_date,
                                       end=target_date + pd.Timedelta(days=1),
                                       freq='3H')[:-1],
                ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)]
            )
        )

        return fig

    except Exception as e:
        print(f"Error in generate_time_series_plot for {feature_key}: {str(e)}")
        return None


def gpmc2ppm(C, T):
    M_c = 0.044
    T_K = T + 273.15
    P = 1 / (9.8692 * 10 ** -6)
    R = 8.31441
    n = C / (M_c * 10 ** 3)
    V = (n * R * T_K) / P
    ppm = V / 10 ** -6
    return ppm


def preprocess_co2_to_ppm(df):
    df = df.copy()
    df['CO2_ref_ppm'] = df.apply(lambda row: gpmc2ppm(row['CO2_ref'], row['Temp_ref']), axis=1)
    return df


def identify_extremee_points(series):
    extreme_points = []
    for i in range(1, len(series) - 1):
        if (series[i] - series[i - 1]) * (series[i + 1] - series[i]) < 0:
            extreme_points.append(i)
    return extreme_points


def extract_features(df, feature):
    # Extract the time series for the target feature
    time_series_target_date = df.set_index('Date')[feature].values
    # Convert time series to 1D array
    time_series_target_date = time_series_target_date.reshape(-1, 1)
    return time_series_target_date


def identify_major_clusters(extreme_points, cluster_labels, min_cluster_size=3, num_largest_clusters=5):
    # Find the indices of clusters with points close to each other
    cluster_indices = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_indices:
            cluster_indices[label] = []
        cluster_indices[label].append(extreme_points[i])

    # Filter clusters with points close to each other
    close_clusters = [cluster for cluster in cluster_indices.values() if len(cluster) >= min_cluster_size]

    # Find the largest clusters with the most points
    largest_clusters = sorted(close_clusters, key=len, reverse=True)[:num_largest_clusters]

    # Flatten the list of largest clusters
    largest_cluster_indices = [index for cluster in largest_clusters for index in cluster]

    return largest_cluster_indices

def detect_sudden_drops(series, threshold):
    drop_points = []
    for i in range(1, len(series)):
        if (series[i - 1] - series[i]) >= threshold:
            drop_points.append(i)
    return drop_points


def find_optimal_threshold(series, num_std_dev=0.5):
    differences = np.diff(series)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    threshold = mean_diff + num_std_dev * std_diff
    return threshold


def detect_trend(data_points, start_value, end_value):
    trends = []
    prev_value = None  # Variable to store the previous value
    prev_trend = None  # Variable to store the previous trend

    # Append start value
    trends.append((data_points[0][0], start_value, "Start Value"))

    for i, (current_time, current_value) in enumerate(data_points):
        # Round the values to two decimal points
        current_value = round(current_value, 2)

        # Calculate the difference between current and previous values
        if prev_value is not None:
            diff = current_value - prev_value
            percentage_change = (current_value - prev_value) / prev_value * 100 if prev_value != 0 else 0
        else:
            diff = 0
            percentage_change = 0

        # Classify the trend based on the rounded difference and percentage change
        if diff > 0.1:  # Adjust threshold as needed
            trend = "sharp increase"
        elif diff > 0.05:  # Adjust threshold as needed
            trend = "exponential increase"
        elif diff > 0:  # Adjust threshold as needed
            trend = "increase"
        elif diff < -0.1:  # Adjust threshold as needed
            trend = "sharp decrease"
        elif diff < -0.05:  # Adjust threshold as needed
            trend = "exponential decrease"
        elif diff < 0:  # Adjust threshold as needed
            trend = "decrease"
        else:
            trend = "no significant change"

        # If the current value is significantly different from the previous one, adjust the trend
        if prev_value is not None:
            if abs(diff) > 10 and prev_trend in ["sharp decrease", "sharp increase"]:
                if diff > 0:
                    trend = "exponential increase"
                else:
                    trend = "exponential decrease"

        # Add the trend to the list
        trends.append((current_time, current_value, trend))

        # Update the previous value and trend
        prev_value = current_value
        prev_trend = trend

    # Append end value
    trends.append((data_points[-1][0], end_value, "End Value"))

    return trends


def calculate_correlations_summarized(df, feature, time_series, feature_name_mapping, drop_points=None, threshold=0.7):
    correlations, correlation_summary = calculate_correlations(df, feature, time_series, feature_name_mapping,
                                                               drop_points)
    significant_correlations = [(feat, corr) for feat, corr in correlations.items() if abs(corr) >= threshold]

    drop_point_correlations = ""
    if drop_points is not None:
        drop_point_correlations = "\n   Key correlations around sudden drops:\n"
        for point in drop_points:
            if point < 0 or point >= len(df):
                continue

            time_index = point
            time_at_drop = time_series[time_index].strftime('%H:%M')
            feature_value_at_drop = df[feature].iloc[time_index]

            drop_point_correlations += (
                f"     - At {time_at_drop}, {feature} dropped to {feature_value_at_drop:.2f}.\n")
            window_size = 10
            start = max(time_index - window_size, 0)
            end = min(time_index + window_size + 1, len(df))

            window_df = df.iloc[start:end]
            window_time_series = time_series[start:end]

            window_correlations, _ = calculate_correlations(window_df, feature, window_time_series,
                                                            feature_name_mapping, drop_points)
            significant_window_correlations = [(feat, corr) for feat, corr in window_correlations.items() if
                                               abs(corr) >= threshold]

            if significant_window_correlations:
                drop_point_correlations += (f"         Correlations observed:\n")
                for feat, corr in significant_window_correlations:
                    drop_point_correlations += f"         * {feat}: Pearson correlation = {corr:.3f}\n"

    return drop_point_correlations + correlation_summary


def create_explanation(df, features_to_analyze, target_date, feature_name_mapping, time_series):
    overall_text_generated = False
    explanation = ""
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d')
    # Check if feature_name_mapping is a dictionary
    if not isinstance(feature_name_mapping, dict):
        raise TypeError("feature_name_mapping should be a dictionary.")

    for feature in features_to_analyze:
        time_series_target_date = df.set_index('Date')[feature].values
        extreme_points = identify_extremee_points(time_series_target_date)

        if len(extreme_points) < 3:
            start_value = time_series_target_date[0]
            end_value = time_series_target_date[-1]
            mapped_feature_name, unit = feature_name_mapping.get(feature, (feature, ""))

            # General Introduction (only generated once)
            if not overall_text_generated:
                explanation += (
                    f"\nOn {target_date.strftime('%B %d, %Y')}, let's explore how the greenhouse conditions "
                    f"changed throughout the day:\n")
                overall_text_generated = True

            explanation += f"\n### {mapped_feature_name} ({feature})\n"

            # Trends Section
            explanation += "\n**Trends**\n"
            explanation += (f"   - At the beginning of the day, the {mapped_feature_name} is {start_value:.2f} {unit}. "
                            f"It remained relatively stable, ending the day at {end_value:.2f} {unit}.\n")

            explanation += calculate_correlations_summarized(df, feature, time_series, feature_name_mapping)

            gradual_changes = capture_gradual_changes(df, feature, time_series)
            if gradual_changes:
                explanation += "\n**Gradual Changes Detected**\n"
                for start_time, end_time, change_type, change_desc in gradual_changes:
                    explanation += f"   - From {start_time} to {end_time}, there is a {change_type} ({change_desc}).\n"

            continue

        start_value = time_series_target_date[extreme_points[0]]
        end_value = time_series_target_date[extreme_points[-1]]
        data_points = [(time_series[0], start_value)] + [(time_series[i], time_series_target_date[i]) for i in
                                                         extreme_points] + [(time_series[-1], end_value)]
        trends = detect_trend(data_points, start_value, end_value)

        optimal_threshold = find_optimal_threshold(time_series_target_date, num_std_dev=2)
        drop_points = detect_sudden_drops(time_series_target_date, optimal_threshold)

        mapped_feature_name, unit = feature_name_mapping.get(feature, (feature, ""))
        if not overall_text_generated:
            explanation += (f"\nOn {target_date.strftime('%B %d, %Y')}, let's explore how the greenhouse conditions "
                            f"changed throughout the day:\n")
            overall_text_generated = True

        explanation += f"\n### {mapped_feature_name} ({feature})\n"

        # Trends Section
        explanation += "\n**Trends**\n"
        explanation += f"   - At the beginning of the day, the {mapped_feature_name} is {start_value:.2f} {unit}."

        major_trends = [trend for trend in trends if
                        trend[2] not in ["Start Value", "End Value", "no significant change"]]
        if not major_trends:
            explanation += (
                f"\n   - The {mapped_feature_name} remained consistent throughout the day, staying at {start_value:.2f} {unit}.\n"
            )
        else:
            previous_value = start_value
            for i, (time, value, trend) in enumerate(trends[1:], start=1):
                previous_time = trends[i - 1][0]
                previous_value = trends[i - 1][1]

                if trend not in ["Start Value", "End Value", "no significant change"]:
                    explanation += (
                        f"\n   - From {previous_time.strftime('%H:%M')} to {time.strftime('%H:%M')}, the {mapped_feature_name} "
                        f"experienced a {trend} from {previous_value:.2f} {unit} to {value:.2f} {unit}."
                    )

        # Correlations Section
        explanation += "\n\n**Correlations**\n"
        explanation += calculate_correlations_summarized(df, feature, time_series, feature_name_mapping)

        # Sudden Drops Section
        if drop_points:
            explanation += "\n\n**Sudden Drops**\n"
            explanation += calculate_correlations_summarized(df, feature, time_series, feature_name_mapping,
                                                             drop_points)
        else:
            explanation += calculate_correlations_summarized(df, feature, time_series, feature_name_mapping)

    return explanation


def capture_gradual_changes(df, feature, time_series, window_size=5):
    gradual_changes = []
    feature_data = df[feature].values
    times = df['Date'].values

    for i in range(0, len(feature_data) - window_size):
        start = i
        end = i + window_size
        segment = feature_data[start:end]
        time_segment = times[start:end]

        # Calculate the mean percentage change over the window
        start_value = segment[0]
        end_value = segment[-1]
        percentage_change = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0

        # Ensure times are formatted correctly
        start_time = time_series[start].strftime('%H:%M')
        end_time = time_series[end - 1].strftime('%H:%M')

        # Adjust thresholds for significance based on percentage change
        if percentage_change > 0.01:  # A small but significant increase
            gradual_changes.append((start_time, end_time, "gradual increase", f"{percentage_change:.2f}% increase"))
        elif percentage_change < -0.01:  # A small but significant decrease
            gradual_changes.append((start_time, end_time, "gradual decrease", f"{percentage_change:.2f}% decrease"))

    return gradual_changes


def calculate_correlations(df, feature, time_series, feature_name_mapping, drop_points=None):
    correlations = {}
    feature_data = df[feature].values

    for other_feature in df.columns:
        if other_feature != feature:
            try:
                other_data = df[other_feature].values
                corr, _ = pearsonr(feature_data, other_data)
                correlations[other_feature] = corr
            except Exception as e:
                print(f"Error calculating Pearson correlation between {feature} and {other_feature}: {e}")

    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    correlation_summary = f"\n   Correlations for {feature}:\n"
    for other_feature, corr in sorted_correlations:
        mapped_feature_name, _ = feature_name_mapping.get(other_feature, (other_feature, ""))
        correlation_summary += (f"     - {mapped_feature_name}: Pearson correlation = {corr:.3f}\n")

    return correlations, correlation_summary


client = setup_openai_client()

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text):
    return model.encode(text)

def process_feature_selection(user_input, term_to_feature):
    features_to_plot = []
    for term, features in term_to_feature.items():
        if term in user_input.lower():
            features_to_plot.extend(features)
    return list(set(features_to_plot))

def preprocess_and_chunk(document_content, selected_features):
    chunks = []
    chunk_embeddings = []
    for feature in selected_features:
        feature_data = extract_feature_data(document_content, feature)
        for chunk in feature_data:
            full_chunk = f"{feature}: {chunk}"
            chunks.append(full_chunk)
            embedding = create_embeddings(full_chunk)
            chunk_embeddings.append(embedding)
    return chunks, chunk_embeddings


def improved_chunking_and_embedding(document_content, selected_features, window_size=200, overlap=50):
    chunks = []
    chunk_embeddings = []
    for feature in selected_features:
        feature_data = extract_feature_data(document_content, feature)
        # Use sliding window approach
        for i in range(0, len(feature_data), window_size - overlap):
            window = feature_data[i:i + window_size]
            full_chunk = f"{feature}: {' '.join(window)}"
            chunks.append(full_chunk)
            embedding = create_embeddings(full_chunk)
            chunk_embeddings.append(embedding)
    return chunks, np.array(chunk_embeddings)

def extract_feature_data(document_content, feature):
    # Find the start of the feature section
    start_index = document_content.find(f"{feature}:")
    if start_index == -1:
        return []  # Feature not found

    # Find the start of the next feature section or the end of the document
    next_feature_index = document_content.find("\n\n", start_index)
    if next_feature_index == -1:
        next_feature_index = len(document_content)

    feature_section = document_content[start_index:next_feature_index]

    # Split the feature section into sentences
    sentences = sent_tokenize(feature_section)
    return sentences

def find_relevant_chunks(question, chunk_embeddings):
    # Convert question to embedding
    question_embedding = create_embeddings(question)

    # Convert chunk_embeddings to numpy array if it's a list
    if isinstance(chunk_embeddings, list):
        chunk_embeddings = np.array(chunk_embeddings)

    # Ensure chunk_embeddings is 2D
    if chunk_embeddings.ndim == 1:
        chunk_embeddings = chunk_embeddings.reshape(1, -1)
    elif chunk_embeddings.ndim == 3:
        chunk_embeddings = chunk_embeddings.squeeze(axis=1)

    # Calculate similarities
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]

    # Get top 3 relevant chunks
    relevant_indices = np.argsort(similarities)[-3:][::-1]

    return relevant_indices.tolist()

def parse_question(question):
    features = {
        "Temp_ref": ["temp", "temperature"],
        "CO2_ref_ppm": ["co2", "carbon dioxide"],
        "Abshum_ref": ["humidity", "moisture"],
        "Bio_ref": ["biomass", "plant growth"],
        "heat_ref": ["heat"],
        "cool_ref": ["cooling"],
        "CO2_inj_ref": ["co2 injection"],
        "outdoor_temp": ["outdoor temperature"],
        "outdoor_humidity": ["outdoor humidity"],
        "outside_radiation": ["outside radiation", "radiation"],
        "co_level": ["co level", "carbon monoxide"]
    }

    categories = {
        "trends": ["trend", "pattern", "overview"],
        "sudden changes": ["drop", "spike", "sudden"],
        "correlation": ["correlation", "relationship"],
        "recommendations": ["recommend", "advise", "suggest"]
    }

    question_lower = question.lower()

    selected_feature = next((key for key, terms in features.items() if any(term in question_lower for term in terms)), None)
    selected_category = next((key for key, terms in categories.items() if any(term in question_lower for term in terms)), "trends")

    return selected_feature, selected_category

def filter_context(context, features, category):
    lines = context.split('\n')
    filtered_lines = []
    current_feature = None
    include_line = False

    for line in lines:
        if line.startswith('###'):
            current_feature = line.strip('# ').lower()
            include_line = any(f.lower() in current_feature for f in features)
        elif line.strip().lower() in ["trends", "sudden drops", "correlation analysis", "recommendations"]:
            include_line = (line.strip().lower() == category.lower() or category.lower() == "all")

        if include_line:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)

def get_prompt_template(category, features):
    base_instructions = """
    IMPORTANT: 
    - Report all values in their original units as they appear in the document. 
    - Do not use placeholder values (like X, Y, A, B). Use the actual numbers from the data.
    - Do not convert units (e.g., °C should remain °C).
    - Provide separate analyses for each requested feature.
    - Only include information directly related to the requested category and features.
    """

    templates = {
        "trends": f"""
        Provide a concise summary of the daily trends for the specified feature(s).
        For each feature, include the following information:
        1. Starting value at the beginning of the day (use the exact value from the data)
        2. Ending value at the end of the day (use the exact value from the data)
        3. Highest and lowest values reached during the day (use exact values from the data)
        4. General pattern of change throughout the day (e.g., gradual increase, fluctuating, etc.)
        {base_instructions}
        """,
        "correlation": f"""
        Focus on the correlations for the specified feature(s).
        For each feature:
        - Mention the most relevant correlations with other features.
        - Explain the relationships in plain text without using numerical values.
        - Highlight any particularly strong or unexpected correlations.
        {base_instructions}
        """,
        "sudden drops": f"""
        Describe any sudden or significant changes for the specified feature(s).
        For each feature, include:
        1. Time of the sudden change
        2. Magnitude of the change
        3. Duration of the change, if applicable
        Focus only on abrupt changes, not general trends.
        {base_instructions}
        """,
        "recommendations": f"""
        Provide brief, actionable advice for managing the specified feature(s).
        For each feature:
        - Base recommendations on the observed trends and changes.
        - Include:
          1. Suggested actions to optimize conditions
          2. Potential impacts on crop yield or energy efficiency
        Do not include technical details or statistical information.
        {base_instructions}
        """
    }

    template = templates.get(category, f"Provide a general analysis of the specified feature(s).\n{base_instructions}")

    feature_list = ", ".join(features)
    return f"{template}\n\nAnalyze the following feature(s) separately: {feature_list}."


def identify_requested_features(user_question, available_features, feature_name_mapping):
    requested_features = set()  # Use a set to avoid duplicates
    question_lower = user_question.lower()

    # Create a mapping of common terms to features
    term_to_feature = {
        'temperature': ['Temp_ref'],
        'temp': ['Temp_ref'],
        'humidity': ['Abshum_ref'],
        'co2': ['CO2_ref_ppm'],
        'carbon dioxide': ['CO2_ref_ppm'],
        'biomass': ['Bio_ref'],
        'plant growth': ['Bio_ref'],
        'heating': ['heat_ref'],
        'cooling': ['cool_ref'],
        'co2 injection': ['CO2_inj_ref'],
        'outdoor temperature': ['outdoor_temp'],
        'outside temperature': ['outdoor_temp'],
        'outdoor humidity': ['outdoor_humidity'],
        'outside humidity': ['outdoor_humidity'],
        'radiation': ['outside_radiation'],
        'co level': ['co_level'],
        'carbon monoxide': ['co_level']
    }

    # Check for each term in the question
    for term, features in term_to_feature.items():
        if term in question_lower:
            for feature in features:
                if feature in available_features:
                    requested_features.add(feature)

    # If no features are found, fall back to checking display names
    if not requested_features:
        for feature, (display_name, unit) in feature_name_mapping.items():
            if feature.lower() in question_lower or display_name.lower() in question_lower:
                requested_features.add(feature)

    # Special case for distinguishing between indoor and outdoor temperature
    if 'Temp_ref' in requested_features and 'outdoor_temp' in requested_features:
        if 'outdoor' in question_lower or 'outside' in question_lower:
            requested_features.remove('Temp_ref')
        else:
            requested_features.remove('outdoor_temp')

    # Special cases for heat_ref, vent_ref, and bio_ref
    special_cases = {
        'heat_ref': ['heating', 'heat'],
        'bio_ref': ['biomass', 'plant growth', 'bio']
    }

    for feature, keywords in special_cases.items():
        if feature in requested_features:
            if not any(keyword in question_lower for keyword in keywords):
                requested_features.remove(feature)

    # If still no features are found, return all available features
    return list(requested_features) if requested_features else available_features


def answer_question(user_question, chunks, chunk_embeddings, overall_explanation, greenhouse_setup, available_features, feature_name_mapping,term_to_feature, weights=None):
    # Debug logging
    print(f"Debug - Chunks: {len(chunks) if isinstance(chunks, list) else 'Invalid chunks'}")
    print(f"Debug - Chunk embeddings shape: {chunk_embeddings.shape if hasattr(chunk_embeddings, 'shape') else 'Invalid embeddings'}")

    # Validate inputs
    if not isinstance(chunks, list):
        return "I apologize, but there seems to be an issue with the data analysis. Could you try selecting your features again?"

    if not isinstance(chunk_embeddings, np.ndarray):
        return "I need to analyze some greenhouse data before I can answer questions. Could you please select some features to look at?"

    if len(chunks) != chunk_embeddings.shape[0]:
        return f"There seems to be a mismatch in the data analysis. Chunks: {len(chunks)}, Embeddings: {chunk_embeddings.shape[0]}"


    # Rest of the function remains the same
    # If weights are not provided, use equal weights
    if weights is None:
        weights = np.ones(len(chunks))

    # Ensure weights are normalized
    weights = weights / np.sum(weights)

    # Find relevant chunks using weighted cosine similarity
    question_embedding = create_embeddings(user_question)
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    weighted_similarities = similarities * weights
    relevant_indices = np.argsort(weighted_similarities)[-3:][::-1]

    relevant_chunks = [chunks[i] for i in relevant_indices]
    context = "\n".join(relevant_chunks)
    constraints_context = """
            We are observing the environmental conditions in a Venlo-type greenhouse with an area of 309 m², equipped with ventilation for air exchange, CO2 injection (23 l/min capacity), heating (110 kW capacity), and cooling (80 kW capacity). The features we are monitoring include temperature, CO2 levels, and relative humidity. We have set point trajectories from an economic optimal control problem aimed at maximizing yield and minimizing operating costs. 
            Our MPC soft constraints are: Temp [18, 26] °C; CO2 [500, 900] ppm; RH [60, 90]%. Hard constraints are: Temp [14, 30] °C; CO2 [100, 1000] ppm; RH [10, 100]%.
        """
    model_context = """
        Greenhouse Model:
        
        1. Variables Influencing Each Other:
           - Temperature (T): Influenced by solar radiation (Qrad), ventilation (uV), heat flux through the cover (Qcov), crop transpiration (Qtrans), heating (uQh), and cooling (uQc).
           - CO2 concentration (C): Influenced by CO2 injection (uC), ventilation (uV), photosynthesis (Cphot), and outdoor CO2 concentration (Cout).
           - Humidity (Ha): Influenced by transpiration (Htrans), ventilation (uV), condensation on the cover (Hcov), heating (Hheat), and cooling (Hcool).
           - Biomass production (B): Influenced by CO2 concentration (C), solar radiation (Qrad), temperature (T), and humidity (Ha).
        
        2. Key Equations and Their Influencing Factors:
           Temperature (T):
           T˙ = (Qsun + Qvent + Qcov + Qtrans + Qheat - Qcool) / kC,gh
           - Qsun: Affected by total radiation transmittance (ktot)
           - Qvent: Affected by ventilation control input (uV), outdoor temperature (Tout), and air heat capacity (kcair)
           - Qcov: Affected by cover heat transfer coefficient (kU), outdoor temperature (Tout)
           - Qtrans: Affected by plant transpiration rate, leaf area index (kLAI), and absolute humidity (Ha)
           - Qheat: Controlled by heating input (uQh)
           - Qcool: Controlled by cooling input (uQc)
        
           CO2 Concentration (C):
           C˙ = (kA,gh / kV,gh) * (Cinj + Cvent - Cphot)
           - Cinj: Controlled by CO2 injection rate (uC)
           - Cvent: Affected by ventilation control input (uV), outdoor CO2 concentration (Cout)
           - Cphot: Affected by radiation (Qrad), CO2 concentration (C), temperature (T), and humidity (Ha)
        
           Absolute Humidity (Ha):
           Ha˙ = (kA,gh / kV,gh) * (Htrans - Hvent - Hcov + Hheat - Hcool)
           - Htrans: Affected by plant transpiration conductance and leaf area index (kLAI)
           - Hvent: Affected by ventilation control input (uV), outdoor humidity (Hout)
           - Hcov: Affected by cover temperature and dew point temperature
           - Hheat: Controlled by heating power (Qheat)
           - Hcool: Affected by cooling pipe parameters and indoor air humidity (Ha)
        
           Biomass Production (B):
           B˙ = (kB,CO2 * ϕCO2) / kA,s
           - ϕCO2 (Photosynthesis Rate): Affected by solar radiation (Qrad), CO2 concentration (C), temperature (T), and humidity (Ha)
        
        3. Greenhouse Control Inputs:
           - uV: Ventilation control
           - uC: CO2 injection control
           - uQh: Heating control
           - uQc: Cooling control
    
        4. Disturbances:
           - Tout: Outdoor temperature
           - Cout: Outdoor CO2 concentration
           - Hout: Outdoor humidity
           - Qrad: Solar radiation
        """
    context += f"\n\nConstraints and Setup:\n{constraints_context}\n\nGreenhouse Model:\n{model_context}"

    requested_features = identify_requested_features(user_question, available_features, feature_name_mapping)

    category = "all" if len(requested_features) > 1 else parse_question(user_question)[1]
    filtered_context = filter_context(overall_explanation, requested_features, category)

    category_prompt = get_prompt_template(category, requested_features)

    is_feature_question = len(requested_features) > 0

    if is_feature_question:
        prompt_template = f"""
            Answer the following question about greenhouse data:
            Question: {user_question}
            
            Greenhouse Setup:
            {greenhouse_setup}
    
            Specific Context (USE THIS INFORMATION FIRST):
            {filtered_context}
    
            Greenhouse Model:
            {model_context}
    
            Instructions:
            {category_prompt}
            Provide a response focusing on the trends of the requested feature(s): {', '.join(requested_features)}.
            You are a highly skilled AI trained in greenhouse and horticulture, with specific knowledge of the greenhouse model and constraints. Your task is to address 
            the question and give a comprehensive response. Focus on the key points that could help a farmer understand the main trends without needing to read the 
            entire dataset.
    
            IMPORTANT:
            - Do not include any correlation values or technical details.
            - Only provide information directly related to the question and available in the context.
            - If multiple features are requested, provide a summary for each feature.
            - Only discuss the features specifically mentioned in the user's question.
            - Consider the relationships, overall greenhouse setup and constraints defined in the greenhouse model when analyzing trends and making recommendations.
            - Base your analysis and explanations on the mathematical model provided, including the equations and relationships between variables.
            - Explain the relationships between variables in simple terms, avoiding technical jargon.
            - Your response should be easily understandable by farmers without a technical background.
            - Do not structure the response with separate segments or bullet points. Instead, provide a flowing, conversational explanation.
            - When discussing relationships between variables, use concrete examples from the provided data to illustrate these relationships.
            - If the question relates to outdoor conditions or their impact, make sure to reference the specific outdoor data provided in the context.
            - Avoid making general statements without backing them up with specific information from the provided context or greenhouse model.
            
            """
    else:
        prompt_template = f"""
            Answer the following question about greenhouse management and horticulture:
            Question: {user_question}
            
            Greenhouse Setup:
            {greenhouse_setup}
    
            Greenhouse Model:
            {model_context}
    
            Instructions:
            You are a highly skilled AI trained in greenhouse management, horticulture, plant science, and Model Predictive Control (MPC) for greenhouses. Your task is to address 
            the question and give a comprehensive response, focusing exclusively on topics related to:
            1. Greenhouse management and operations
            2. Plant science and horticulture
            3. Model Predictive Control (MPC) in greenhouse settings
            4. Environmental control in greenhouses (temperature, humidity, CO2, lighting, etc.)
            5. Crop yield optimization in controlled environments
    
            IMPORTANT:
            - If the question is not directly related to the above topics, politely redirect the conversation back to greenhouse and horticulture-related subjects.
            - Provide practical, actionable information that would be useful for greenhouse operators or horticulturists.
            - Avoid using technical terms or jargon. Explain concepts in simple, everyday language.
            - If specific data is not available, provide general best practices or principles related to the question.
            - Maintain a focus on practical aspects of greenhouse management and plant cultivation.
            - Consider the overall greenhouse setup and constraints defined in the greenhouse model when providing explanations or recommendations.
            - Base your explanations on the mathematical model provided, but explain relationships in simple, non-technical terms.
            - When discussing control strategies or environmental factors, refer to the specific variables in the model using everyday language.
            - Your response should be in a conversational tone, easily understandable by farmers without a technical background.
            - Do not structure the response with separate segments or bullet points. Instead, provide a flowing, conversational explanation.
            - When discussing relationships between variables, use concrete examples from the provided data to illustrate these relationships.
            - If the question relates to outdoor conditions or their impact, make sure to reference the specific outdoor data provided in the context.
            - Avoid making general statements without backing them up with specific information from the provided context or greenhouse model.
            
            """

    evaluator = LLMResponseEvaluator()

    initial_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in greenhouse data analysis, horticulture, and Model Predictive Control for greenhouses."},
            {"role": "user", "content": prompt_template}
        ]
    )

    initial_evaluation = evaluator.evaluate_response(
        initial_response.choices[0].message.content,
        overall_explanation,
        user_question
    )

    # Additional evaluation layer
    evaluation_prompt = f"""
        Given the initial response to the question "{user_question}":
    
        {initial_response.choices[0].message.content}
    
        Now, refine this response based on the following criteria:
        1. Analyze the response in light of the greenhouse mathematical model:
        {model_context}
        
        2. Explain how the observed trends relate to the mathematical model, focusing on:
           - The key equations influencing the relevant features
           - The relationships between variables as defined in the model
           - How control inputs and disturbances might have affected the observed trends
        
        3. Provide a more in-depth explanation of why the observed trends occurred, using the model as a framework.
        
        4. If the initial response lacks specific data or observations, add relevant information from the following context:
        {filtered_context}
        
        5. Ensure the refined response:
           - Explains the relationships between variables in simple, non-technical terms that a farmer can easily understand
           - Avoids using correlation values or technical jargon
           - Provides a flowing, conversational explanation without separate segments or bullet points
           - Uses concrete examples from the provided data to illustrate relationships between variables
           - Focuses on providing actionable insights that a greenhouse operator can use in their daily operations
            

        Initial evaluation scores:
        ROUGE-L: {initial_evaluation['rouge_score']}
        BERTScore: {initial_evaluation['bert_score']}
        Relevance: {initial_evaluation['relevance']}
        
        Refined response:
        """


    refined_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in greenhouse data analysis, horticulture, and Model Predictive Control for greenhouses."},
            {"role": "user", "content": evaluation_prompt}
        ]
    )
    refined_evaluation = evaluator.evaluate_response(
        refined_response.choices[0].message.content,
        overall_explanation,
        user_question
    )

    # Compare evaluations and choose the best response
    if (refined_evaluation['rouge_score'] + refined_evaluation['bert_score'] + refined_evaluation['relevance']) > \
            (initial_evaluation['rouge_score'] + initial_evaluation['bert_score'] + initial_evaluation['relevance']):
        final_response = refined_response.choices[0].message.content
        final_evaluation = refined_evaluation
    else:
        final_response = initial_response.choices[0].message.content
        final_evaluation = initial_evaluation

    # Log the evaluation results
    print(f"Initial Evaluation Scores:")
    print(f"ROUGE-L: {initial_evaluation['rouge_score']}")
    print(f"BERTScore: {initial_evaluation['bert_score']}")
    print(f"Relevance: {initial_evaluation['relevance']}")

    # Log the evaluation results
    print(f"Final Evaluation Scores:")
    print(f"ROUGE-L: {final_evaluation['rouge_score']}")
    print(f"BERTScore: {final_evaluation['bert_score']}")
    print(f"Relevance: {final_evaluation['relevance']}")

    # Save the final evaluation to the metrics
    evaluator.update_metrics(
        final_evaluation['rouge_score'],
        final_evaluation['bert_score'],
        final_evaluation['relevance'],
        None  # Human feedback can be added here if available
    )

    return final_response


def handle_feedback_change(feedback_key):
    feedback = st.session_state.get(feedback_key)
    if feedback:
        try:
            # Get the relevant question and answer
            last_user_message = next((msg for msg in reversed(st.session_state.messages)
                                      if msg["role"] == "user"), None)
            last_assistant_message = next((msg for msg in reversed(st.session_state.messages)
                                           if msg["role"] == "assistant"), None)

            question = last_user_message["content"] if last_user_message else ""
            answer = last_assistant_message["content"] if last_assistant_message else ""

            # Get user type from session state
            user_type = st.session_state.user_info.get('role', 'unknown') if hasattr(st.session_state, 'user_info') else 'unknown'
            st.session_state[feedback_key] = feedback
            # Initialize feedback evaluator if not exists
            if 'feedback_evaluator' not in st.session_state:
                st.session_state.feedback_evaluator = FeedbackEvaluator('feedback_metrics.json')

            # Add feedback
            success = st.session_state.feedback_evaluator.add_feedback(
                feedback,
                user_type,
                question=question,
                answer=answer
            )

            if success:
                st.success(f"Thank you for your feedback!")
                time.sleep(1)  # Display success message for 1 second
                st.empty()  # Clear the success message
            else:
                st.warning("Feedback recorded, but there was an issue saving it.")
                time.sleep(1)  # Display warning for 1 second
                st.empty()
            st.rerun()
        except Exception as e:
            logging.error(f"Error handling feedback: {e}")
            st.error("There was an error processing your feedback.")



def load_data():
    df = pd.read_csv('HC_ref_163_2510.csv')
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True)

    # Add this line to verify the date range in your DataFrame
    print(f"Date range in DataFrame: {df['Date'].min()} to {df['Date'].max()}")

    # Apply transformations
    df['heat_ref'] = df['heat_ref'] * 110000
    df['cool_ref'] = df['cool_ref'] * 80000
    df['CO2_inj_ref'] = df['CO2_inj_ref'] * 0.7792

    return df


def initialize_session_state():
    if 'user_info' not in st.session_state:
        welcome_form()

    defaults = {
        'chunks': [],
        'selected_features': [],
        'messages': [{"role": "assistant", "content": "Welcome to our Smart Greenhouse Chatbot! On which date would you like to see your plants' growth?","avatar": "sprout.gif"}],
        'current_stage': 'welcome',
        'analysis_complete': False,
        'chunk_embeddings': [],
        'greenhouse_setup': """
        We are observing the environmental conditions in a Venlo-type greenhouse with an area of 309 m², 
        equipped with ventilation for air exchange, CO2 injection (23 l/min capacity), 
        heating (110 kW capacity), and cooling (80 kW capacity). The features we are monitoring include 
        temperature, CO2 levels, and relative humidity. We have set point trajectories from an economic 
        optimal control problem aimed at maximizing yield and minimizing operating costs. 
        Our MPC soft constraints are: Temp [18, 26] °C; CO2 [500, 900] ppm; RH [60, 90]%. 
        Hard constraints are: Temp [14, 30] °C; CO2 [100, 1000] ppm; RH [10, 100]%.
        """,
        'all_features_data': [],
        'overall_explanation': "",
        'displayed_plots': {},
        'detailed_explanations': [],
        'trends_dict': {},
        'plots': {},
        'chunk_weights': None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Welcome to our Smart Greenhouse Chatbot! On which date would you like to see your plants' growth?","avatar": "sprout.gif"
        })
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 'welcome'
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'chunk_embeddings' not in st.session_state:
        st.session_state.chunk_embeddings = []
    if 'greenhouse_setup' not in st.session_state:
        st.session_state.greenhouse_setup = """
            We are observing the environmental conditions in a Venlo-type greenhouse with an area of 309 m², 
            equipped with ventilation for air exchange, CO2 injection (23 l/min capacity), 
            heating (110 kW capacity), and cooling (80 kW capacity). The features we are monitoring include 
            temperature, CO2 levels, and relative humidity. We have set point trajectories from an economic 
            optimal control problem aimed at maximizing yield and minimizing operating costs. 
            Our MPC soft constraints are: Temp [18, 26] °C; CO2 [500, 900] ppm; RH [60, 90]%. 
            Hard constraints are: Temp [14, 30] °C; CO2 [100, 1000] ppm; RH [10, 100]%.
            """
    if 'all_features_data' not in st.session_state:
        st.session_state.all_features_data = []
    if 'overall_explanation' not in st.session_state:
        st.session_state.overall_explanation = ""
    if 'feedback_system' not in st.session_state or len(st.session_state.chunks) != len(
            st.session_state.feedback_system.chunk_weights):
        st.session_state.feedback_system = FeedbackSystem(len(st.session_state.chunks))
    if 'feedback_evaluator' not in st.session_state:
        st.session_state.feedback_evaluator = FeedbackEvaluator('feedback_metrics.json')
    if 'displayed_plots' not in st.session_state:
        st.session_state.displayed_plots = {}
    if 'detailed_explanations' not in st.session_state:
        st.session_state.detailed_explanations = []
    if st.session_state.chunks and st.session_state.chunk_weights is None:
        st.session_state.chunk_weights = np.ones(len(st.session_state.chunks))

def process_feature_data(feature, df_processed, time_series, target_date_str, feature_name_mapping):
    try:
        # Convert target_date to datetime if it's a string
        if isinstance(target_date_str, str):
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        elif isinstance(target_date_str, datetime):
            pass
        else:
            raise ValueError(f"Unexpected type for target_date: {type(target_date_str)}")
        time_series_target_date = df_processed[feature].values
        extreme_points = identify_extremee_points(time_series_target_date)

        # Check if there are enough extreme points for clustering
        if len(extreme_points) < 3:
            explanation = create_explanation(df_processed, [feature], target_date, feature_name_mapping, time_series)
            st.session_state.detailed_explanations.append(explanation)
            st.session_state.all_features_data.append(f"{feature}: {explanation}")
            return

        # Trend detection
        data_points = [(time_series[0], time_series_target_date[0])] + \
                      [(time_series[i], time_series_target_date[i]) for i in extreme_points] + \
                      [(time_series[-1], time_series_target_date[-1])]

        trends = detect_trend(data_points, time_series_target_date[0], time_series_target_date[-1])
        st.session_state.trends_dict[feature] = trends
        start_value = time_series_target_date[extreme_points[0]]
        end_value = time_series_target_date[extreme_points[-1]]
        trends = detect_trend(data_points, start_value, end_value)
        st.session_state.trends_dict[feature] = trends

        # Step 2: Determine Optimal Number of Clusters using Silhouette Score
        silhouette_scores = []
        n_samples = len(extreme_points)
        max_clusters = min(n_samples - 1, 10)  # Ensure max_clusters is at most n_samples - 1

        if n_samples < 3:
            print(f"Not enough extreme points ({n_samples}) for clustering. Skipping...")


        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(np.array(extreme_points).reshape(-1, 1))
            if len(np.unique(cluster_labels)) > 1:  # Check if there are more than one cluster
                silhouette_avg = silhouette_score(np.array(extreme_points).reshape(-1, 1), cluster_labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(-1)  # Invalid silhouette score for single cluster

        # Choose the optimal number of clusters based on the maximum Silhouette Score
        optimal_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 because we started from k=2

        # Step 3: Cluster Adjacent Change Points using the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(np.array(extreme_points).reshape(-1, 1))

        # Find the optimal threshold
        optimal_threshold = find_optimal_threshold(time_series_target_date, num_std_dev=2)

        # Detect sudden drops using the optimal threshold
        drop_points = detect_sudden_drops(time_series_target_date, optimal_threshold)

        # Identify major clusters with points close to each other
        major_cluster_indices = identify_major_clusters(extreme_points, cluster_labels)

        explanation = create_explanation(df_processed, [feature], target_date_str, feature_name_mapping,
                                         time_series)
        st.session_state.detailed_explanations.append(explanation)
        st.session_state.all_features_data.append(f"{feature}: {explanation}")

        st.session_state.overall_explanation = "\n".join(st.session_state.detailed_explanations)
        st.session_state.chunks, st.session_state.chunk_embeddings = improved_chunking_and_embedding(
            st.session_state.overall_explanation,
            st.session_state.selected_features
        )
        st.session_state.chunk_weights = np.ones(len(st.session_state.chunks))
    except Exception as e:
        print(f"Error processing feature {feature}: {str(e)}")
        st.session_state.detailed_explanations.append(f"Error processing {feature}: {str(e)}")
        st.session_state.all_features_data.append(f"{feature}: Error occurred during processing")

def display_feedback(message_index):
    feedback_key = f"feedback_{message_index}"

    # Initialize feedback state if not exists
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = None

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.session_state[feedback_key] is None:
            feedback = st.radio(
                "Was this response helpful?",
                options=("Yes", "No", "Partially"),
                key=feedback_key,
                index=None,
                on_change=handle_feedback_change,
                args=(feedback_key,)
            )
        else:
            time.sleep(8)
            st.empty()


def process_user_input(user_input, df, term_to_feature, feature_name_mapping):
    response = ""
    message_placeholder = st.empty()
    full_response = ""
    show_feedback = False
    try:
        # Date processing
        if 'selected_date' not in st.session_state or user_input.replace('-', '').isdigit():
            date_formats = ['%d %B', '%d %b', '%d-%m', '%d%m', '%B %d', '%b %d']
            detected_date = None

            for fmt in date_formats:
                try:
                    detected_date = datetime.strptime(user_input, fmt)
                    detected_date = detected_date.replace(year=2011)
                    break
                except ValueError:
                    continue

            if detected_date:
                st.session_state.selected_date = detected_date
                df_processed = df[df['Date'].dt.date == detected_date.date()]
                if len(df_processed) == 0:
                    response = f"No data available for {detected_date.strftime('%d %B, %Y')}. Please choose another date."
                else:
                    df_processed = preprocess_co2_to_ppm(df_processed)
                    st.session_state.df_processed = df_processed
                    response = f"I've set the date to {detected_date.strftime('%d %B, %Y')}. Would you like to see plots of the greenhouse data for this day? (Yes/No)"
                    st.session_state.awaiting_plot_confirmation = True
            else:
                response = "I couldn't understand that date format. Please provide a date like '26-05', '26 May', or '2605'."

        # Plot confirmation
        elif st.session_state.get('awaiting_plot_confirmation', False):
            if user_input.lower() == 'yes':
                response = "Great! What features would you like to see plots for? You can choose from temperature, humidity, CO2, biomass, heating, cooling, CO2 injection, outdoor temperature, outdoor humidity, radiation, or CO level."
                st.session_state.awaiting_feature_selection = True
                st.session_state.awaiting_plot_confirmation = False
            elif user_input.lower() == 'no':
                response = "Alright. What would you like to know about the greenhouse data for this day?"
                st.session_state.awaiting_plot_confirmation = False
            else:
                response = "I didn't understand that. Please answer 'Yes' if you want to see plots, or 'No' if you don't."

        # Feature selection
        elif st.session_state.get('awaiting_feature_selection', False):
            features_to_plot = process_feature_selection(user_input, term_to_feature)

            if features_to_plot:
                st.session_state.selected_features = features_to_plot
                st.session_state.analysis_complete = True

                # Generate explanations and embeddings
                st.session_state.trends_dict = {}
                st.session_state.detailed_explanations = []
                st.session_state.all_features_data = []

                selected_date_str = st.session_state.selected_date.strftime('%Y-%m-%d')
                time_series = pd.date_range(start=selected_date_str + ' 00:00:00',
                                            end=selected_date_str + ' 23:55:00', freq='5T')

                for feature in features_to_plot:
                    process_feature_data(feature, st.session_state.df_processed, time_series,
                                         selected_date_str, feature_name_mapping)

                st.session_state.overall_explanation = "\n".join(st.session_state.detailed_explanations)
                st.session_state.chunks, st.session_state.chunk_embeddings = improved_chunking_and_embedding(
                    st.session_state.overall_explanation,
                    list(feature_name_mapping.keys())
                )
                st.session_state.chunk_weights = np.ones(len(st.session_state.chunks))

                plots = []
                for feature in features_to_plot:
                    fig = generate_time_series_plot(st.session_state.df_processed, feature,
                                                    st.session_state.selected_date, feature_name_mapping)
                    if fig is not None:
                        plots.append({"feature": feature, "plot": fig})

                response = f"Here are the plots for the requested features: {', '.join(features_to_plot)}. You can now ask questions about the data."
                st.session_state.current_plots = plots
                # Add plots to the message
                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "avatar": "sprout.gif"
                }
                st.session_state.messages.append(assistant_message)
                for plot_info in plots:
                    try:
                        st.plotly_chart(plot_info["plot"], use_container_width=True)
                    except Exception as e:
                        print(f"Error plotting {plot_info['feature']}: {str(e)}")

                plot_message = {
                    "role": "assistant",
                    "content": response,
                    "avatar": "sprout.gif",
                    "show_feedback": True,
                    "has_plots": True
                }
                st.session_state.messages.append(plot_message)

                # Prompt the user to ask a question
                follow_up_message = {
                    "role": "assistant",
                    "content": "What would you like to know about this data? You can ask about any feature or aspect of the greenhouse.",
                    "avatar": "sprout.gif",
                    "show_feedback": False
                }
                st.session_state.messages.append(follow_up_message)

                st.session_state.awaiting_feature_selection = False

                print(f"Debug - Features to plot: {features_to_plot}")
                print(f"Debug - Plots generated: {[plot['feature'] for plot in plots]}")
            else:
                response = "I didn't recognize any feature names. Please specify which features you'd like to see plots for (e.g., temperature, humidity, CO2)."

        # Question answering
        elif (st.session_state.get('df_processed') is not None and
              st.session_state.get('chunks') is not None and
              st.session_state.get('chunk_embeddings') is not None and
              len(st.session_state.get('chunk_embeddings')) > 0):
            all_features = list(feature_name_mapping.keys())

            # Check if the question is about a feature that wasn't initially selected
            requested_feature = None
            for term, features in term_to_feature.items():
                if term.lower() in user_input.lower():
                    requested_feature = features[0]
                    break

            if requested_feature and requested_feature not in st.session_state.selected_features:
                # Process the new feature
                selected_date_str = st.session_state.selected_date.strftime('%Y-%m-%d')
                time_series = pd.date_range(start=selected_date_str + ' 00:00:00',
                                            end=selected_date_str + ' 23:55:00', freq='5T')
                process_feature_data(requested_feature, st.session_state.df_processed, time_series,
                                     selected_date_str, feature_name_mapping)

                # Update the overall explanation and embeddings
                st.session_state.overall_explanation = "\n".join(st.session_state.detailed_explanations)
                st.session_state.chunks, st.session_state.chunk_embeddings = improved_chunking_and_embedding(
                    st.session_state.overall_explanation,
                    list(feature_name_mapping.keys())
                )
                st.session_state.chunk_weights = np.ones(len(st.session_state.chunks))

            if answer := answer_question(user_input, st.session_state.chunks,
                                         st.session_state.chunk_embeddings,
                                         st.session_state.overall_explanation,
                                         st.session_state.greenhouse_setup,
                                         all_features,
                                         feature_name_mapping, term_to_feature,
                                         weights=st.session_state.chunk_weights if st.session_state.chunk_weights is not None else None):

                # Simulate streaming
                for chunk in answer.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)

                message_placeholder.markdown(answer)
                response = answer
                show_feedback = True

        else:
            response = "I encountered an error while processing your request. Could you please rephrase your question?"

    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(f"Debug - Error details: {traceback.format_exc()}")
        response = error_message

    # Add response to messages if not already added
    if response and not any(msg["content"] == response for msg in st.session_state.messages[-2:]):
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "avatar": "sprout.gif",
            "show_feedback": show_feedback
        })
        display_feedback(len(st.session_state.messages) - 1)
    st.rerun()


def monitor_page():
    apply_custom_css()
    st.markdown("<h1 style='text-align: center;'>🌿 Smart Greenhouse Chatbot 🌱</h1>", unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Initialize session states
    if 'user_info' not in st.session_state:
        welcome_form()
        return

    initialize_session_state()

    # Define feature mapping
    term_to_feature = {
        'temperature': ['Temp_ref'],
        'temp': ['Temp_ref'],
        'humidity': ['Abshum_ref'],
        'co2': ['CO2_ref_ppm'],
        'carbon dioxide': ['CO2_ref_ppm'],
        'biomass': ['Bio_ref'],
        'plant growth': ['Bio_ref'],
        'heating': ['heat_ref'],
        'cooling': ['cool_ref'],
        'co2 injection': ['CO2_inj_ref'],
        'outdoor temperature': ['outdoor_temp'],
        'outside temperature': ['outdoor_temp'],
        'outdoor humidity': ['outdoor_humidity'],
        'outside humidity': ['outdoor_humidity'],
        'radiation': ['outside_radiation'],
        'co level': ['co_level'],
        'carbon monoxide': ['co_level']
    }

    df = pd.read_csv('HC_ref_163_2510.csv')
    df.columns = df.columns.str.strip().str.replace('"', '')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True)

    # Apply transformations
    df['heat_ref'] = df['heat_ref'] * 110000
    df['cool_ref'] = df['cool_ref'] * 80000
    df['CO2_inj_ref'] = df['CO2_inj_ref'] * 0.7792

    feature_name_mapping = {
        'Temp_ref': ('Temperature', '°C'),
        'Abshum_ref': ('Relative Humidity', '%'),
        'CO2_ref_ppm': ('CO2', 'ppm'),
        'Bio_ref': ('Biomass', 'kg/m²'),
        'heat_ref': ('Heating Reference', 'kWh'),
        'cool_ref': ('Cooling Reference', 'kWh'),
        'CO2_inj_ref': ('CO2 Injection Reference', 'units'),
        'outdoor_temp': ('Outdoor Temperature', '°C'),
        'outdoor_humidity': ('Outdoor Humidity', '%'),
        'outside_radiation': ('Outside Radiation', 'W/m²'),
        'co_level': ('Outside CO2 Level', 'ppm')
    }

    # Chat interface
    st.write("### Greenhouse Data Analysis Chat")
    for i, message in enumerate(st.session_state.messages):
        if message and "role" in message:  # Ensure the message is not None and has the "role" key
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])
                # Display plots if available
                if message.get("has_plots", False) and hasattr(st.session_state, 'current_plots'):
                    for plot_info in st.session_state.current_plots:
                        try:
                            st.plotly_chart(plot_info["plot"], use_container_width=True)
                        except Exception as e:
                            print(f"Error plotting {plot_info['feature']}: {str(e)}")

                # Show feedback option if applicable
                if message["role"] == "assistant" and message.get("show_feedback", False):
                    display_feedback(i)

    if user_input := st.chat_input("How can I help you with the greenhouse data, give feedback for each response?"):
        user_avatar = "leaves.gif"
        st.session_state.messages.append({"role": "user", "content": user_input, "avatar": user_avatar})

        with st.chat_message("user", avatar=user_avatar):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="sprout.gif"):
            process_user_input(user_input, df, term_to_feature, feature_name_mapping)

    save_user_data(user_info=st.session_state.user_info, chat_messages=st.session_state.messages)

# Main function
def main():
    
    if 'page' not in st.session_state:
        st.session_state.page = "project_explanation"

    if 'nda_agreed' not in st.session_state:
        st.session_state.nda_agreed = True  # Set NDA agreement to true by default

    if st.session_state.page == "project_explanation":
        project_explanation_page()
    elif st.session_state.page == "nda":
        nda_page()
    elif st.session_state.page == "welcome_form":
        if not st.session_state.nda_agreed:
            st.error("You must agree to the NDA before proceeding.")
            st.session_state.page = "nda"
            st.rerun()
        welcome_form()
    elif st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "monitor":
        if not st.session_state.get('user_info'):
            st.error("Please provide your information before accessing the monitor.")
            st.session_state.page = "welcome_form"
            st.rerun()
        if not st.session_state.nda_agreed:
            st.error("You must agree to the NDA before accessing the monitor.")
            st.session_state.page = "nda"
            st.rerun()
        monitor_page()

if __name__ == "__main__":
    main()
