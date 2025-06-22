import os
import time
from datetime import datetime
from dateparser.search import search_dates
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS



# 1. Configurer votre clé API Mistral
api_key = os.getenv('MISTRAL_AI_KEY')

# 2. Charger les embeddings Mistral
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

# 3. Charger la base vectorielle FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 4. Charger le modèle de chat Mistral
llm = ChatMistralAI(model="mistral-small", api_key=api_key)

# 5. Définir le prompt
prompt_template = PromptTemplate.from_template("""
Tu es un assistant culturel spécialisé dans les événements en région Occitanie. Tu parles toujours en français.

Tu as accès à l'historique de la conversation avec l'utilisateur. Si l'utilisateur te pose une question sur des informations personnelles (comme son prénom ou sa ville), réponds uniquement à cette question **sans jamais proposer d'événements**.

Tu ne proposes des événements culturels que lorsque l'utilisateur te le demande clairement.

Historique de la conversation :
{chat_history}

Contexte :
{context}

Question de l'utilisateur :
{question}

Si tu ne trouves pas d'information dans la mémoire ou les documents, dis-le poliment sans inventer.
""")



# 6. Créer la mémoire conversationnelle (fenêtre de 3 échanges)
memory = ConversationBufferWindowMemory(
    k=3,
    return_messages=True,
    memory_key="chat_history"
)

# 7. Construire la chaîne RAG avec mémoire
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={
        "prompt": prompt_template,
        "document_variable_name": "context"
    }
)

# 8. Boucle de chat
print("🤖 Bienvenue dans le chatbot culturel Occitanie avec mémoire ! Posez votre question (ou tapez 'exit' pour quitter)\n")

while True:
    user_input = input("Vous : ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print(" À bientôt !")
        break

    try:
        response = qa_chain.invoke({"question": user_input})
        result = response.get("answer", "").strip()

        if not result or "aucun événement" in result.lower():
            print(" Désolé, je n'ai trouvé aucun événement correspondant à votre recherche.")
        else:
            print(f"\nAssistant : {result}\n")
    except Exception as e:
        if "429" in str(e):
            print(" Trop de requêtes envoyées à l'API Mistral. Attendez quelques secondes et réessayez.")
            time.sleep(5)
        else:
            print(" Une erreur est survenue :", e)
