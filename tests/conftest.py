import os

# ai.py builds its clients at import time; dummy values keep tests offline.
for k, v in {
    "LLM_BASE_URL": "http://localhost:0/v1",
    "LLM_API_KEY": "test",
    "CHAT_MODEL": "test",
    "EMBED_MODEL": "test",
    "SEARCH_TOOL_NAME": "test",
}.items():
    os.environ.setdefault(k, v)
