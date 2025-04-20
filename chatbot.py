import os
import time
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI



# 1. Configurer votre clé API Mistral
api_key = os.getenv('MISTRAL_AI_KEY')

# 2. Charger les embeddings Mistral
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

# 3. Charger la base vectorielle FAISS
vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True  # nécessaire car on utilise pickle
)

# 4. Charger le modèle de chat Mistral
llm = ChatMistralAI(model="mistral-small", api_key=api_key)

# 5. Définir le prompt avec context et question
prompt_template = PromptTemplate.from_template("""
Tu es un assistant culturel spécialisé dans les événements en région Occitanie.

En te basant sur les informations suivantes, propose des recommandations d'événements pertinentes à l'utilisateur.

Contexte : 
{context}

Question de l'utilisateur : 
{question}

Si tu ne trouves pas d'événement correspondant, informe l'utilisateur de manière courtoise.
""")

# 6. Construire la chaîne QA avec récupération via FAISS
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={
        "prompt": prompt_template,
        "document_variable_name": "context"  # 👈 Obligatoire pour injecter le contexte dans le prompt
    }
)

# 7. Boucle de chat
print("🤖 Bienvenue dans le chatbot culturel Occitanie ! Posez votre question (ou tapez 'exit' pour quitter)\n")

while True:
    user_input = input("Vous : ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("👋 À bientôt !")
        break

    try:
        response = qa_chain.invoke({"query": user_input})
        context = response.get("result", "").strip()  # Récupère le contexte des résultats

        # Vérification du contexte avant d'afficher
        if not context or "aucun événement" in context.lower():
            print("❌ Désolé, je n'ai trouvé aucun événement correspondant à votre recherche.")
        else:
            print(f"\nAssistant : {context}\n")
    except Exception as e:
        # Gestion des erreurs (par exemple, 429 pour trop de requêtes envoyées)
        if "429" in str(e):
            print("🚦 Trop de requêtes envoyées à l'API Mistral. Attendez quelques secondes et réessayez.")
            time.sleep(5)  # Attente de 5 secondes avant de réessayer
        else:
            print("❌ Une erreur est survenue :", e)
