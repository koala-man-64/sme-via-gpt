import pytest
import re
from unittest.mock import MagicMock, patch
from app import ConversationManager

def test_conversation_id_validation_valid():
    """Verify that a valid UUID allows processing to proceed (mocking storage)."""
    cm = ConversationManager()
    valid_id = "550e8400-e29b-41d4-a716-446655440000"
    
    # patch the internal _get_blob_client to avoid actual azure calls
    with patch.object(cm, '_get_blob_client') as mock_get_client:
        mock_blob_client = MagicMock()
        mock_blob_client.exists.return_value = False # Simulating session not found, which is fine, just checking it doesn't return early due to validation
        mock_get_client.return_value = mock_blob_client
        
        history, blobs = cm.load_session(valid_id)
        
        # It should have called _get_blob_client because validation passed
        mock_get_client.assert_called_once_with(valid_id)
        assert history == []
        assert blobs == []

def test_conversation_id_validation_invalid_chars():
    """Verify that invalid characters are rejected."""
    cm = ConversationManager()
    invalid_id = "../../../etc/passwd"
    
    with patch.object(cm, '_get_blob_client') as mock_get_client:
        history, blobs = cm.load_session(invalid_id)
        
        # Should NOT have called _get_blob_client
        mock_get_client.assert_not_called()
        assert history == []
        assert blobs == []

def test_conversation_id_validation_injection():
    """Verify that potential command injection chars are rejected."""
    cm = ConversationManager()
    invalid_id = "id; rm -rf /"
    
    with patch.object(cm, '_get_blob_client') as mock_get_client:
        history, blobs = cm.load_session(invalid_id)
        mock_get_client.assert_not_called()

def test_conversation_id_validation_simple_alphanumeric():
    """Verify simple alphanumeric IDs work (backward compatibility if needed)."""
    cm = ConversationManager()
    valid_id = "chat123"
    
    with patch.object(cm, '_get_blob_client') as mock_get_client:
        mock_blob_client = MagicMock()
        mock_blob_client.exists.return_value = False
        mock_get_client.return_value = mock_blob_client
        
        cm.load_session(valid_id)
        mock_get_client.assert_called_once()
