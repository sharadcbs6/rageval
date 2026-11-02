import os
from dotenv import load_dotenv
load_dotenv(".env")
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
models = client.models.list()
for m in models.data:
    ctx = getattr(m, "context_window", "?")
    print(f"{m.id:<40} context={ctx}")
