import os
import csv
import logging
from datetime import datetime, date
import locale
from functools import lru_cache
from duckduckgo_search import DDGS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from pathlib import Path

# -------- LOGGING --------
TODAY = datetime.now().date()
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f"logs/chatbot_{TODAY}.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
today_str = date.today().strftime("%A %d %B %Y")

# -------- Fallback Web --------
@lru_cache(maxsize=50)
def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="fr-fr", safesearch="Moderate", max_results=3)
            output = ""
            for r in results:
                output += f"- {r['title']} ({r['href']})\n{r['body']}\n\n"
            return output if output else "Aucun résultat trouvé."
    except Exception as e:
        logging.error(f"Erreur DuckDuckGo : {e}")
        return "⚠️ La recherche en ligne a échoué. Réessayez plus tard."

# -------- API Mistral & Embeddings --------
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    logging.error("Clé API Mistral manquante.")
    raise ValueError("La clé API Mistral est absente.")

embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

def charger_index_faiss(chemin_index: str, embeddings):
    index_path = Path(chemin_index)
    if not index_path.exists() or not (index_path / "index.faiss").exists():
        logging.error(f"Index FAISS introuvable dans : {chemin_index}")
        return None

    try:
        return FAISS.load_local(
            chemin_index,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logging.error(f"Erreur de chargement FAISS : {e}")
        return None

vectorstore = charger_index_faiss("faiss_index", embeddings)
if vectorstore is None:
    raise RuntimeError("Index FAISS introuvable ou erreur de chargement")

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

def get_bot_response(question: str, user_location: dict = None) -> str:
    """
    Retourne la réponse du chatbot pour une question donnée,
    en utilisant la localisation passée (optionnelle).
    """
    global today_str

    parsed_question = f"Réponds toujours en français. {question} (Nous sommes le {today_str})"
    if user_location and user_location.get("city"):
        parsed_question += f" (Je suis à {user_location['city']})"

    try:
        response = qa_chain.invoke({"question": parsed_question})
        result = response.get("answer", "").strip()

        mots_cles_fallback = [
            "je n'ai pas", "aucune information", "aucun événement",
            "pas d'informations", "je ne trouve pas"
        ]
        besoin_recherche_web = (not result) or any(m in result.lower() for m in mots_cles_fallback)

        if besoin_recherche_web:
            web_result = search_web(question)
            return web_result

        return result

    except Exception as e:
        logging.exception("Erreur lors de la réponse :")
        return "❌ Une erreur est survenue lors du traitement de votre demande."
