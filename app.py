import streamlit as st
from datetime import datetime, date
from chatbot_core import get_bot_response
from geo import get_user_location

# Détection de la localisation (stockée une fois)
if "location" not in st.session_state:
    location = get_user_location()
    st.session_state.location = location

# Interface
st.set_page_config(page_title="Chatbot Culturel Occitanie", page_icon="🎭")
st.title("🎭 Chatbot Culturel Occitanie")
st.markdown("Posez vos questions sur les événements culturels en région Occitanie. 📍")

# Initialisation historique messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Champ de saisie utilisateur
user_input = st.chat_input("Posez votre question ici...")

# Traitement de la question et réponse
if user_input:
    location = st.session_state.location
    # Appel à la fonction du core, en passant la localisation si besoin
    response = get_bot_response(user_input, location)
    # Mémoriser les échanges
    st.session_state.chat_history.append(("Vous", user_input))
    st.session_state.chat_history.append(("Assistant", response))

# Afficher l’historique du chat
for role, msg in st.session_state.chat_history:
    with st.chat_message(role.lower()):
        st.markdown(msg)
