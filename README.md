# 🎯 Job Match AI Agent

An intelligent agent that analyzes your resume against any Job Description, scores the match, identifies gaps, and rewrites your resume sections to align with the role.

## 🏗️ Architecture

```
app.py (Streamlit UI)
    │
    ├── src/resume_parser.py   → Parse .docx / .pdf resume into text
    ├── src/rag_engine.py      → Embed resume into FAISS, retrieve relevant chunks
    ├── src/agent.py           → LangGraph agent (5-node pipeline)
    │       ├── Node 1: parse_jd      → Extract structured JD requirements
    │       ├── Node 2: score_match   → Score resume vs JD (0-100)
    │       ├── Node 3: gap_analysis  → Identify missing/weak skills
    │       ├── Node 4: rewrite       → Rewrite summary + bullets
    │       └── Node 5: report        → Compile full report
    └── src/exporter.py        → Export rewritten content to .docx
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd job_match_agent
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Azure OpenAI

Copy `.env.template` to `.env` and fill in your credentials:

```bash
cp .env.template .env
```

Edit `.env`:
```
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

### 3. Run

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

## 🎮 How to Use

1. Enter your Azure OpenAI credentials in the **sidebar**
2. **Upload** your resume (.docx or .pdf)
3. **Paste** the full Job Description
4. Click **🚀 Analyze & Match**
5. Review your match score, gaps, and rewritten content
6. **Download** the aligned resume .docx

## 📂 Project Structure

```
job_match_agent/
├── app.py                  # Streamlit frontend
├── requirements.txt        # Dependencies
├── .env.template           # Azure config template
├── .env                    # Your actual config (git-ignored)
├── src/
│   ├── resume_parser.py    # Resume text extraction
│   ├── rag_engine.py       # FAISS vector store + retrieval
│   ├── agent.py            # LangGraph 5-node agent
│   └── exporter.py         # .docx + .txt export
├── outputs/                # Generated files saved here
└── README.md
```

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Azure OpenAI (GPT-4o) |
| Embeddings | Azure OpenAI (text-embedding-ada-002) |
| Vector Store | FAISS (local, in-memory) |
| Agent Framework | LangGraph |
| LLM Orchestration | LangChain |
| Frontend | Streamlit |
| Resume Parsing | python-docx, pypdf |
| Export | python-docx |

## 🔧 Extending the Project

**Ideas to add next:**
- [ ] Scrape JD directly from LinkedIn/Indeed URL
- [ ] Track multiple JD applications in a SQLite DB
- [ ] Add skill-learning recommendations with links
- [ ] Email the report to yourself automatically
- [ ] Compare multiple JDs side by side
- [ ] Add a cover letter generator node to LangGraph

## 📝 Notes

- The agent does NOT fabricate experience — it only rewords and emphasizes what's already in your resume
- All files are saved locally in the `outputs/` folder
- Your API key is never stored — only used in the current session
