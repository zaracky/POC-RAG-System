import os
import time
import requests
import csv
from datetime import datetime
from dateparser import parse as parse_date
from duckduckgo_search import DDGS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from geo import get_user_location

#  Recherche web DuckDuckGo
def search_web(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, region="fr-fr", safesearch="Moderate", max_results=3)
        output = ""
        for r in results:
            output += f"- {r['title']} ({r['href']})\n{r['body']}\n\n"
        return output if output else "Aucun résultat trouvé."

#  Date du jour
TODAY = datetime.now().date()

#  Configurer votre clé API Mistral
api_key = os.getenv('MISTRAL_AI_KEY')


# Embeddings + vectordb
embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Modèle + prompt
llm = ChatMistralAI(model="mistral-small", api_key=api_key)

prompt_template = PromptTemplate.from_template("""
Tu es un assistant culturel spécialisé dans les événements en région Occitanie. Tu parles toujours en français.

Tu as accès à l'historique de la conversation avec l'utilisateur. Si l'utilisateur te pose une question sur des informations personnelles (comme son prénom ou sa ville), réponds uniquement à cette question sans jamais proposer d'événements.

Tu ne proposes des événements culturels que lorsque l'utilisateur te le demande clairement.

Historique de la conversation :
{chat_history}

Contexte :
{context}

Question de l'utilisateur :
{question}

Si tu ne trouves pas d'information dans la mémoire ou les documents, dis-le poliment sans inventer.
""")

memory = ConversationBufferWindowMemory(
    k=3,
    return_messages=True,
    memory_key="chat_history"
)

retriever = vectorstore.as_retriever()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={
        "prompt": prompt_template,
        "document_variable_name": "context"
    }
)

#  Localisation
user_location = get_user_location()
if not user_location or not user_location.get("city"):
    print(" Localisation introuvable")
else:
    print(f" Localisation détectée automatiquement : {user_location['city']}, {user_location['region']}")

#  Chat CLI
print(" Bienvenue dans le chatbot culturel Occitanie avec recherche web ! Tapez 'exit' pour quitter\n")

log_file = f"csv/chatbot_logs_{TODAY}.csv"
os.makedirs("csv", exist_ok=True)

while True:
    user_input = input("Vous : ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print(" À bientôt !")
        feedback = input("Avez-vous trouvé cela utile ? (o/n) : ").lower()
        if feedback in ["o", "y"]:
            print("Merci pour votre retour positif !")
        else:
            print("Merci pour votre retour, nous améliorerons l'expérience.")
        break

    parsed_question = user_input
    if user_location and user_location.get("city"):
        parsed_question += f" (Je suis à {user_location['city']})"

    try:
        response = qa_chain.invoke({"question": parsed_question})
        result = response.get("answer", "").strip()

        if not result or "aucun événement" in result.lower():
            print(" Je cherche en ligne, un instant...")
            web_result = search_web(user_input)
            print(f"\n Résultats web :\n{web_result}\n")
        else:
            print(f"\nAssistant : {result}\n")

        with open(log_file, mode="a", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().isoformat(), user_input, result])

    except Exception as e:
        print(" Une erreur est survenue :", e)
