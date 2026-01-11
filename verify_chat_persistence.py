
import os
import json
import uuid
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

# Config
ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")
CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

def verify():
    print(f"Checking access to {ACCOUNT}/{CONTAINER}...")
    
    if CONN_STR:
        bsc = BlobServiceClient.from_connection_string(CONN_STR)
    else:
        from azure.identity import DefaultAzureCredential
        url = f"https://{ACCOUNT}.blob.core.windows.net"
        bsc = BlobServiceClient(url, credential=DefaultAzureCredential())

    conv_id = str(uuid.uuid4())
    blob_name = f"chats/{conv_id}.json"
    
    print(f"Simulating chat persistence for ID: {conv_id}")
    
    # Simulate what ConversationManager does
    client = bsc.get_blob_client(container=CONTAINER, blob=blob_name)
    
    payload = {
        "history": [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"}
        ],
        "selected_blobs": ["test_doc.pdf"],
        "last_updated": "2023-01-01T00:00:00Z"
    }
    
    client.upload_blob(json.dumps(payload), overwrite=True)
    print("Upload successful.")
    
    # Verify download
    print("Verifying download...")
    data = json.loads(client.download_blob().readall())
    
    assert data["history"][0]["content"] == "Hello world"
    assert "test_doc.pdf" in data["selected_blobs"]
    
    print("Verification Passed! Blob content matches.")
    
    # Clean up
    client.delete_blob()
    print("Cleanup successful.")

if __name__ == "__main__":
    try:
        verify()
    except Exception as e:
        print(f"Verification FAILED: {e}")
