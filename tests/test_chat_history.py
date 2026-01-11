
import unittest
from unittest.mock import MagicMock, patch
import json
from app import ConversationManager

class TestConversationManager(unittest.TestCase):
    def setUp(self):
        self.mock_blob_service_client = MagicMock()
        self.mock_container_client = MagicMock()
        self.mock_blob_client = MagicMock()
        
        self.mock_blob_service_client.get_container_client.return_value = self.mock_container_client
        self.mock_container_client.get_blob_client.return_value = self.mock_blob_client
        
        # Patch the global helper in app.py
        self.patcher = patch('app.azure_blob_service_client', return_value=self.mock_blob_service_client)
        self.mock_abs = self.patcher.start()
        
        self.cm = ConversationManager(max_turns=2) # Small max_turns for easier truncation testing

    def tearDown(self):
        self.patcher.stop()

    def test_get_history_empty(self):
        # Setup: Blob does not exist
        self.mock_abs.return_value.get_blob_client.return_value.exists.return_value = False
        
        history = self.cm.get_history("new_conv")
        self.assertEqual(history, [])

    def test_get_history_existing(self):
        # Setup: Blob exists with data
        mock_data = {
            "history": [{"role": "user", "content": "hi"}],
            "selected_blobs": ["a.txt"]
        }
        self.mock_abs.return_value.get_blob_client.return_value.exists.return_value = True
        self.mock_abs.return_value.get_blob_client.return_value.download_blob.return_value.readall.return_value = json.dumps(mock_data).encode('utf-8')

        history = self.cm.get_history("existing_conv")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["content"], "hi")

    def test_add_turn_and_persistence(self):
        # Setup: Blob does not exist initially
        self.mock_abs.return_value.get_blob_client.return_value.exists.return_value = False
        
        self.cm.add_turn("conv1", "hello", "world", selected_blobs=["doc1"])
        
        # Verify upload called
        upload_call = self.mock_abs.return_value.get_blob_client.return_value.upload_blob.call_args
        self.assertIsNotNone(upload_call)
        
        uploaded_data = json.loads(upload_call[0][0])
        self.assertEqual(len(uploaded_data["history"]), 2) # user + assistant
        self.assertEqual(uploaded_data["history"][0]["content"], "hello")
        self.assertEqual(uploaded_data["history"][1]["content"], "world")
        self.assertEqual(uploaded_data["selected_blobs"], ["doc1"])

    def test_truncation(self):
        # Setup: start with empty
        self.mock_abs.return_value.get_blob_client.return_value.exists.return_value = False
        
        # Add 3 turns (6 messages). Max turns is 2 (4 messages).
        self.cm.add_turn("conv1", "1", "1a", [])
        self.cm.add_turn("conv1", "2", "2a", [])
        self.cm.add_turn("conv1", "3", "3a", [])
        
        # Check last upload
        upload_call = self.mock_abs.return_value.get_blob_client.return_value.upload_blob.call_args
        uploaded_data = json.loads(upload_call[0][0])
        
        self.assertEqual(len(uploaded_data["history"]), 4) # 2 * max_turns
        self.assertEqual(uploaded_data["history"][0]["content"], "2") # First one truncated

if __name__ == '__main__':
    unittest.main()
