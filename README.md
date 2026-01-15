# BT Bot

RAG system for analyzing Ben Thompson's Stratechery content with intelligent retrieval and structured reasoning.

## Architecture

- **Two-tier retrieval**: Distillations (high-level analysis) + Full-text chunks (precision)
- **Knowledge distillation**: Structured extraction of thesis, incentives, predictions, counterarguments
- **Copyright protection**: Strict quotation limits and refusal behaviors
- **Evaluation harness**: Custom metrics for retrieval quality and analytical reasoning

## Tech Stack

- **Backend**: FastAPI, Python
- **Database**: Supabase (PostgreSQL + pgvector)
- **AI**: OpenAI (embeddings, generation, judge), Claude (analysis agents)
- **Frontend**: React (coming soon)

## Setup

1. Clone the repository
2. Create virtual environment: `python3 -m venv venv`
3. Activate: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your API keys
6. Run database setup SQL in Supabase
7. Start server: `python -m uvicorn app.main:app --reload`

## Project Status

- [x] Project setup
- [x] Database schema
- [ ] Email ingestion pipeline
- [ ] Distillation generation
- [ ] Two-tier retrieval
- [ ] Evaluation system
- [ ] React dashboard