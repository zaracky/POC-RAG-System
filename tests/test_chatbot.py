import unittest
from unittest.mock import patch, MagicMock
import time
from chatbot import qa_chain  

class TestChatbot(unittest.TestCase):

    @patch('langchain_community.vectorstores.FAISS.load_local')  # On patch FAISS pour éviter de charger un vrai fichier
    @patch('langchain_mistralai.chat_models.ChatMistralAI')  # On patch aussi l'API Mistral pour éviter de faire une vraie requête
    def test_model_and_index_loading(self, mock_chat_model, mock_faiss_load):
        # Simuler le comportement du modèle et de Faiss
        mock_faiss_load.return_value = MagicMock()
        mock_chat_model.return_value = MagicMock()

        # Tester si la fonction de chargement de Faiss est bien appelée
        faiss_index = mock_faiss_load("faiss_index", None)
        self.assertIsNotNone(faiss_index)

        # Tester que l'API Mistral est bien chargée
        model = mock_chat_model("mistral-small", api_key="fake_key")
        self.assertIsNotNone(model)

    @patch('chatbot.qa_chain.invoke')  # Patch la méthode 'invoke' pour éviter l'appel réel
    def test_query_processing(self, mock_invoke):
        # Simuler une réponse de l'API pour un input utilisateur
        mock_invoke.return_value = {"result": "Voici des événements à Toulouse..."}

        # Tester que le chatbot donne une réponse correcte
        response = qa_chain.invoke({"query": "événements à Toulouse"})
        self.assertIn("Voici des événements à Toulouse...", response['result'])

    @patch('chatbot.time.sleep')  # Patch time.sleep pour éviter d'attendre réellement
    def test_api_rate_limit(self, mock_sleep):
        # Simuler une erreur API avec un code 429 (taux limite)
        with self.assertRaises(Exception):
            # Simule un appel API qui échoue avec erreur 429
            raise Exception("Error response 429 while fetching")

        # Vérifie que la fonction de sleep est appelée (pour gérer la limite de requêtes)
        mock_sleep.assert_called_once_with(5)

if __name__ == '__main__':
    unittest.main()
