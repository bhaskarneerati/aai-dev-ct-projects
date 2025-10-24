# RAG Assistant with Structured Logging

## 🧠 Overview
The **RAG Assistant** is a modular Retrieval-Augmented Generation (RAG) system built for research and document-based Q&A. It integrates multiple LLM providers (OpenAI, Groq, Google Gemini) with a Chroma vector database and supports structured logging for clarity and traceability.

---

## 📁 Project Structure
```bash
├── data/                   # Source text documents (.txt)
├── src/
    ├── chats/              # session chat transcripts 
    ├── chroma_db/          # persistant embedded vector database
    ├── logs/               # session logs
    ├── app.py              # Main RAG assistant CLI application
    ├── vectordb.py         # Vector database wrapper using Chroma & HuggingFace embeddings
    ├── logger.py           # Custom logger with colored output, file logging, and traceback support
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
└── README.md               # This guide
```

---

## ⚙️ Features
### Core Capabilities
- Multi-LLM support (OpenAI, Groq, Google Gemini)
- Vector search using **ChromaDB** and **HuggingFace embeddings**
- Automatic text chunking for improved retrieval
- Modular, testable architecture

### Enhanced Logging System
- Separate **session logs** (\`*_log.txt\`) and **chat transcripts** (\`*_chat.txt\`)
- Timestamp format: \`YYYYMMDD_HHMMSS_mmm\` (IST timezone)
- Prompts and model responses recorded in log files for traceability

---

## 🧩 Setup Instructions
### 1. Clone Repository
```bash
git clone https://github.com/bhaskarneerati/aai-dev-ct-projects.git
cd rag-assistant
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\\Scripts\\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy \`.env.example\` to \`.env\` and fill in your API keys:
```bash
OPENAI_API_KEY=your_openai_key
or
GROQ_API_KEY=your_groq_key
or
GOOGLE_API_KEY=your_google_key
```

You may also specify model names (optional):
```bash
OPENAI_MODEL=gpt-4o-mini
GROQ_MODEL=llama-3.1-8b-instant
GOOGLE_MODEL=gemini-2.0-flash
```

---

## 🚀 Usage
### Running the Assistant
```bash
python src/app.py
```

Ask questions interactively:
```
Question> What are the key findings in AI safety research?
```

The assistant responds with a synthesized answer and source document list.

### Log and Chat Files
Each run creates two files under \`logs/\`:
```
20251023_153045_123_log.txt    # detailed logs with prompts & responses
20251023_153045_123_chat.txt   # user and assistant dialogue with IST timestamps
```

---

## 🧱 Architecture
### Components
| Module | Responsibility |
|--------|----------------|
| \`app.py\` | CLI entry point, orchestrates document loading, retrieval, and LLM invocation |
| \`vectordb.py\` | Handles chunking, embedding, and semantic search using ChromaDB |
| \`logger.py\` | Manages colored terminal logs, file logging |

### Execution Flow
1. Initialize LLM client and VectorDB  
2. Load text documents from \`/data\`  
3. Embed and store in Chroma collection  
4. Accept user queries  
5. Retrieve top relevant chunks  
6. Construct research-style prompt  
7. Invoke LLM, log full prompt & result  
8. Display concise output with color-coded feedback  

---

## 📊 Logging Example

Excerpt from log file:
```
2025-10-23 15:30:45.123 IST [INFO] Document ‘research.txt’ ingested into VectorDB
2025-10-23 15:31:12.456 IST [ERROR] Failed to embed ‘missing.txt’: FileNotFoundError
```

Excerpt from chat file:
```
[2025-10-23 15:32:12 IST] User: What is RAG?
[2025-10-23 15:32:13 IST] Assistant: Retrieval-Augmented Generation (RAG) combines vector search with LLMs to answer questions based on documents.
```

---

## 🧰 Dependencies
- \`langchain\`, \`langchain_openai\`, \`langchain_groq\`, \`langchain_google_genai\`
- \`chromadb\`
- \`torch\`
- \`huggingface_hub\`
- \`rich\`
- \`python-dotenv\`

---

## 🔒 Environment & Security
- API keys are loaded via \`.env\` (never hard-coded).  
- Log files may contain prompts and model outputs—store securely.  
- Local persistence uses \`./chroma_db\`.

---

## 👩‍💻 Author & Maintainers
**Bhaskar** — AI Agent Builder & Architect

#### Refernces:
Ready Tensor’s git hub repo - [rt-aaidc-project1-template](https://github.com/readytensor/rt-aaidc-project1-template).

> _Designed for clarity, observability, and extensibility — a foundation for reliable RAG systems._
