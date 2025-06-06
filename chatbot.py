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

# 3. Chargement de l'index FAISS
vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 4. Modèle Mistral
llm = ChatMistralAI(model="mistral-small", api_key=api_key)

# 5. Ajout de la date du jour dans le prompt
aujourdhui = datetime.today().strftime("%Y-%m-%d")

prompt_template = PromptTemplate.from_template(f"""
Tu es un assistant culturel spécialisé dans les événements en région Occitanie.
Aujourd'hui, nous sommes le {aujourdhui}.

L'utilisateur pose une question à propos d'événements dans une période spécifique.
NE FOURNIS que des événements DONT LA DATE DE DÉBUT EST POSTÉRIEURE OU ÉGALE à la date d'aujourd'hui, et idéalement dans la période précisée.
Si aucun événement ne correspond, indique-le simplement.
 NE FOURNIS PAS d'événements passés même s'ils sont proches ou similaires.

Contexte :
{{context}}

Question de l'utilisateur :
{{question}}
""")


# 6. Création de la chaîne QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={
        "prompt": prompt_template,
        "document_variable_name": "context"
    }
)

# 7. Fonction pour enrichir les requêtes temporelles

def enrichir_question(question):
    from dateparser.search import search_dates
    import dateparser
    from datetime import datetime, timedelta

    # Gestion manuelle de "ce week-end"
    if "ce week-end" in question.lower() or "ce weekend" in question.lower():
        today = datetime.today()
        samedi = today + timedelta((5 - today.weekday()) % 7)  # prochain samedi
        dimanche = samedi + timedelta(days=1)
        question = question.lower().replace("ce week-end", f"entre {samedi.strftime('%Y-%m-%d')} et {dimanche.strftime('%Y-%m-%d')}")
        question = question.replace("ce weekend", f"entre {samedi.strftime('%Y-%m-%d')} et {dimanche.strftime('%Y-%m-%d')}")

    # Forcer dateparser à chercher vers le futur
    settings = {
        'PREFER_DATES_FROM': 'future',
        'RELATIVE_BASE': datetime.now()
    }

    dates_detectees = search_dates(question, settings=settings, languages=['fr'])
    date_annonce = ""

    if dates_detectees:
        for expr, date in dates_detectees:
            question = question.replace(expr, date.strftime("%Y-%m-%d"))
        if len(dates_detectees) >= 1:
            date_annonce = f"\n(période ciblée : du {dates_detectees[0][1].strftime('%Y-%m-%d')}"
            if len(dates_detectees) > 1:
                date_annonce += f" au {dates_detectees[1][1].strftime('%Y-%m-%d')})"
            else:
                date_annonce += ")"
    return question + date_annonce



# 8. Boucle de chat
print("\U0001f916 Bienvenue dans le chatbot culturel Occitanie ! Posez votre question (ou tapez 'exit' pour quitter)\n")

while True:
    user_input = input("Vous : ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("👋 À bientôt !")
        break

    try:
        question_enrichie = enrichir_question(user_input)
        response = qa_chain.invoke({"query": question_enrichie})
        context = response.get("result", "").strip()

        if not context or "aucun événement" in context.lower():
            print("❌ Désolé, je n'ai trouvé aucun événement correspondant à votre recherche.")
        else:
            print(f"\nAssistant : {context}\n")
    except Exception as e:
        if "429" in str(e):
            print("🚦 Trop de requêtes envoyées à l'API Mistral. Attendez quelques secondes et réessayez.")
            time.sleep(5)
        else:
            print("❌ Une erreur est survenue :", e)
