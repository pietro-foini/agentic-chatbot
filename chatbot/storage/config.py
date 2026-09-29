from google.cloud import storage

from chatbot.settings import S

storage_client = storage.Client(project=S.PROJECT_ID)
