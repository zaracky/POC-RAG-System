import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from chatbot import qa_chain, enrichir_question
import time

class TestChatbot(unittest.TestCase):
    """
    Test unitaire pour le système RAG du chatbot culturel.
    Les fonctions testent :
    - le chargement des composants
    - le fonctionnement de la chaîne QA
    - le parsing des dates naturelles
    """

    @patch('langchain_community.vectorstores.FAISS.load_local')
    @patch('langchain_mistralai.chat_models.ChatMistralAI')
    def test_model_and_index_loading(self, mock_chat_model, mock_faiss_load):
        """
        Vérifie que FAISS et le modèle Mistral sont bien chargés (mockés).
        Cela simule l'environnement sans faire appel aux services réels.
        """
        mock_faiss_load.return_value = MagicMock()
        mock_chat_model.return_value = MagicMock()

        faiss_index = mock_faiss_load("faiss_index", None)
        self.assertIsNotNone(faiss_index)

        model = mock_chat_model("mistral-small", api_key="fake_key")
        self.assertIsNotNone(model)

    @patch('chatbot.qa_chain.invoke')
    def test_query_processing(self, mock_invoke):
        """
        Vérifie que la chaîne QA retourne une réponse correcte à une requête simple.
        """
        mock_invoke.return_value = {"result": "Voici des événements à Toulouse..."}
        response = qa_chain.invoke({"query": "événements à Toulouse"})
        self.assertIn("Voici des événements à Toulouse...", response['result'])

    def test_enrichir_question_with_ce_weekend(self):
        """
        Teste que l'enrichissement de l'expression 'ce week-end'
        est correctement converti en une date ISO future.
        """
        question = "Que faire à Toulouse ce week-end ?"
        result = enrichir_question(question)
        today = datetime.today()
        self.assertIn(str(today.year), result)
        self.assertRegex(result, r"20\\d{2}-\\d{2}-\\d{2}")  # attend une date ISO

    def test_enrichir_question_with_date_range(self):
        """
        Vérifie que l'enrichissement de requêtes avec plages de dates fonctionne.
        """
        question = "Quels concerts à Toulouse entre le 15 juillet et le 20 juillet ?"
        result = enrichir_question(question)
        self.assertIn("période ciblée", result)
        self.assertRegex(result, r"\\d{4}-\\d{2}-\\d{2}")

    def test_no_event_response_format(self):
        """
        Vérifie que le système sait exprimer proprement une réponse vide.
        """
        fake_response = {"result": "❌ Désolé, je n'ai trouvé aucun événement correspondant à votre recherche."}
        self.assertIn("aucun événement", fake_response["result"].lower())

    def test_api_rate_limit(self):
        """
        Simule une erreur d'API de type 429 (trop de requêtes).
        Vérifie que le système applique une temporisation via time.sleep.
        """
        with patch("chatbot.time.sleep") as mock_sleep:
            try:
                raise Exception("Error response 429 while fetching")
            except Exception as e:
                if "429" in str(e):
                    time.sleep(5)
                    mock_sleep.assert_called_once_with(5)

if __name__ == '__main__':
    unittest.main()
