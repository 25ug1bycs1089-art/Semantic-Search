# Document Q&A

A local-first Streamlit app for asking questions about PDF, DOCX, and TXT files. It indexes passages with semantic embeddings, answers from retrieved passages, shows document/page or section citations, and stores conversations in SQLite.

## Run locally

1. Install Python 3.11 or newer.
2. Install the project dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

3. Create a Groq API key in the [Groq console](https://console.groq.com/keys).
4. Configure the key before launching the app. In PowerShell:

   ```powershell
   $env:GROQ_API_KEY = "your-groq-api-key"
   ```

   Alternatively, put it in a `.env` file in the project folder:

   ```dotenv
   GROQ_API_KEY=your-groq-api-key
   ```

   Do not add a trailing semicolon. The app loads `.env` automatically, and `.env` is excluded from Git. Or create `.streamlit/secrets.toml`:

   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   ```

5. Start the app:

   ```powershell
   py -m streamlit run streamlit_app.py
   ```

Install the requirements before starting the app, using the same Python installation for both commands. If an upload reports that `sentence_transformers` is missing, run `py -m pip install -r requirements.txt` from the project folder and restart Streamlit.

Answers are generated through Groq's hosted chat API, so questions require an internet connection and a valid Groq API key. The default model is `openai/gpt-oss-120b`; set `GROQ_MODEL` to use another model supported by your Groq account.

The app uses the official OpenAI Python SDK pointed at Groq's OpenAI-compatible API (`https://api.groq.com/openai/v1`), with streaming and `reasoning_effort="medium"`. If Groq returns Cloudflare error 1010, its edge network is blocking the current network/IP; changing SDKs cannot bypass that restriction, so try a network allowed by Groq or contact Groq support.

Document retrieval remains local. When the `sentence-transformers/all-MiniLM-L6-v2` model is already cached, the app uses it for semantic search without contacting Hugging Face. If it is not cached or Hugging Face is unreachable, indexing switches to local keyword matching so uploads still work offline; this fallback is less effective for paraphrases. Groq only replaces answer generation and does not provide document embeddings in this app. The current question, recent conversation context, and retrieved passages are sent to Groq to generate an answer; uploaded files and the full local index are not sent. Conversation history and document indexes are stored in `.data/document_qa.sqlite3`; set `QA_DB_PATH` to use another SQLite file. Remove documents from the sidebar when they should no longer be searchable.

## Tests

```powershell
py -m pip install pytest
py -m pytest
```
