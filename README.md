# 🤖 Persona Support Agent

An advanced AI-powered Customer Support Agent that identifies user personas and provides personalized, context-aware responses using Retrieval-Augmented Generation (RAG).

## 🚀 Key Features
- **Persona Classification:** Detects whether a user is a 'Frustrated User', 'Technical Expert', or 'Business Executive'.
- **Context-Aware Responses:** Uses RAG pipeline to fetch relevant business policies, API guides, and support documents.
- **Dynamic Tone Adaptation:** Adjusts the AI's response tone based on the detected user persona.
- **Built with:** Python, Streamlit, LangChain, ChromaDB, HuggingFace, and **Groq API**.

## 📂 Project Structure
- `app.py`: Main Streamlit application interface.
- `src/classifier.py`: Logic to detect user intent/persona.
- `src/rag_pipeline.py`: Document loading, chunking, and Vector DB setup.
- `src/generator.py`: LLM logic for generating responses using **Groq**.
- `src/config.py`: Configuration settings.
- `data/`: Contains knowledge base documents (.txt, .md, .pdf).

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd persona-support-agent