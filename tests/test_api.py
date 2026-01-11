
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app import app, ConversationManager

client = TestClient(app)

class TestAPI(unittest.TestCase):
    def setUp(self):
        # Patch dependencies
        self.patcher_abs = patch('app.azure_blob_service_client')
        self.mock_abs = self.patcher_abs.start()
        
        self.patcher_openai = patch('app.openai_client')
        self.mock_openai = self.patcher_openai.start()
        
        self.patcher_convo = patch('app.conversation_manager')
        self.mock_convo = self.patcher_convo.start()

    def tearDown(self):
        self.patcher_abs.stop()
        self.patcher_openai.stop()
        self.patcher_convo.stop()

    def test_get_models(self):
        response = client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIn("default", data)
        self.assertTrue(len(data["models"]) > 0)

    def test_chat_endpoint_happy_path(self):
        # Mock RAG index readiness
        with patch('app.rag.is_ready', return_value=True), \
             patch('app.rag.search', return_value=[]):
            
            # Mock OpenAI response
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="Mocked Answer"))]
            self.mock_openai.return_value.chat.completions.create.return_value = mock_resp

            payload = {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-4o",
                "conversation_id": "test-id"
            }
            response = client.post("/api/chat", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["choices"][0]["message"]["content"], "Mocked Answer")
            
            # Verify ConversationManager called
            self.mock_convo.add_turn.assert_called_once()

    def test_upload_endpoint(self):
        # Mock Blob Client
        mock_container = MagicMock()
        mock_blob = MagicMock()
        self.mock_abs.return_value.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob
        
        # Mock File
        files = {'file': ('test.txt', b'content', 'text/plain')}
        response = client.post("/api/upload", files=files)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("filename", response.json())
        
        # Verify upload
        mock_blob.upload_blob.assert_called_once()

if __name__ == "__main__":
    unittest.main()
