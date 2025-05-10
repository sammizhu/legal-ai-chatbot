# LegalAI Chatbot

A fullstack Retrieval-Augmented Generation (RAG) chatbot for legal data, leveraging judge biographical and reassignment data, legal opinions, and modern LLMs. This project enables users to ask legal questions and receive contextually grounded answers, referencing real cases and judge profiles.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Data Preparation & Embedding Generation](#data-preparation--embedding-generation)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [End-User Documentation](#end-user-documentation)
- [Development & Deployment Notes](#development--deployment-notes)

---

## Features
- **Data ingestion:** Reads judge bios, reassignment data, and legal opinions from `.dta`, `.csv`, and `.json` files.
- **Embeddings:** Generates and saves vector embeddings for all datasets using TogetherAI.
- **RAG Pipeline:** Retrieves relevant cases and judge profiles for any user query and uses an LLM to generate an answer.
- **Fullstack:** FastAPI backend serving a chat UI (`chat.html`).
- **Stateless API:** `/chat` endpoint for programmatic or UI-based Q&A.

---

## Project Structure

```
legal-ai-chatbot/
├── backend/
│   └── main.py         # FastAPI backend
├── data/               # All processed data and embeddings
│   ├── ...
├── judge_bio.py        # Judge bio data processing
├── generate_embeddings.py # Embedding generation script
├── rag_pipeline.py     # RAG logic and retrieval
├── chat.html           # Frontend chat UI
├── requirements.txt    # Python dependencies
├── .env                # API keys and secrets
└── README.md           # This file
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/legal-ai-chatbot.git
   cd legal-ai-chatbot
   ```

2. **Install Python dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up your API key:**
   - Create a `.env` file in the project root with:
     ```
     TOGETHER_API_KEY=your_actual_together_api_key_here
     ```

4. **Prepare your data:**
   - Place all source data files (judge bios, legal opinions, reassignment CSVs) in the appropriate locations as described in the scripts.

---

## Data Preparation & Embedding Generation

1. **Process Judge Bio Data:**
   - Run `judge_bio.py` to convert `.dta` bios to `data/judge_profiles.json`.

2. **Convert Reassignment CSVs to JSON:**
   - Use your CSV-to-JSON script or process to ensure all reassignment datasets are in `data/` as `.json` files.

3. **Generate Embeddings:**
   - Run:
     ```bash
     python generate_embeddings.py
     ```
   - This script reads all required JSONs and outputs `.npy` embedding files in `data/`.

---

## Running the Backend

1. **Start the FastAPI server:**
   ```bash
   uvicorn backend.main:app --reload
   ```
   - The backend will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - The `/chat` endpoint accepts POST requests with `{ "question": "..." }` and returns `{ "answer": "..." }`
   - The root URL `/` serves the chat UI (`chat.html`).

---

## Running the Frontend

- Simply visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.
- The chat UI will load and allow you to interact with the chatbot.

---

## End-User Documentation

### How to Use the Chatbot
1. **Open your browser** and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) (or your deployed URL).
2. **Type your legal question** in the input box and press Send.
3. **Wait for the response.** The bot will retrieve relevant cases and judges, then generate a grounded answer using the LLM.
4. **Review the answer.** The response may reference real cases, judges, and provide context-aware legal information.

### Example Questions
- "What happens if you change a judge in a case?"
- "Who was Judge Hatfield and what cases did he preside over?"
- "What is a recess appointment?"

### Troubleshooting
- If you see `undefined` or no answer, check that the backend is running and the `/chat` endpoint is reachable.
- If you see `{"detail":"Not Found"}` at `/`, the backend is running but the chat UI is not being served—ensure `chat.html` is present.
- For embedding errors, ensure you have run `generate_embeddings.py` and all `.npy` files exist in `data/`.

---

## Development & Deployment Notes
- **Backend expects embeddings and data in `data/` directory.**
- **Deployment:** If deploying (e.g., Render, Railway), ensure the startup command points to `backend.main:app` and `chat.html` is in the project root.
- **Large Files:** Some data files may be large; consider using Git LFS or cloud storage for production.
- **API Key Security:** Never commit your `.env` file or API keys to public repositories.

---

## License
MIT License. See [LICENSE](LICENSE) for details.
