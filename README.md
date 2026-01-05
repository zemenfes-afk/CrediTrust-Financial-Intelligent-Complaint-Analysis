# 🏦 CrediTrust Financial: Intelligent Complaint Analysis Chatbot

**A RAG-powered AI assistant for analyzing consumer financial complaints using Retrieval-Augmented Generation.**

## 📖 Project Overview

The **CrediTrust Intelligent Complaint Analysis** system is designed to help financial analysts and customer support teams understand consumer grievances efficiently.

Instead of manually searching through thousands of rows in Excel, this **RAG (Retrieval-Augmented Generation)** chatbot allows users to ask natural language questions like *"What are the common issues with Student Loans?"*. The system retrieves real-world complaint narratives from a vector database and synthesizes a factual, evidence-backed answer using a local Large Language Model (LLM).

---

## 🚀 Key Features

* **🧠 RAG Pipeline:** Combines semantic search with generative AI to answer questions based strictly on retrieved data.
* **🔍 Semantic Search:** Uses `sentence-transformers/all-MiniLM-L6-v2` to understand the *meaning* of complaints, not just keyword matching.
* **🤖 Privacy-First AI:** Runs entirely offline using **Google's Flan-T5 Base**, ensuring no financial data leaves your local machine.
* **💬 Interactive UI:** Features a user-friendly chat interface built with **Gradio**.
* **📊 Smart Citations:** The chatbot provides "Evidence" for its answers by citing specific Complaint IDs and narratives used to generate the response.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Orchestration:** LangChain
* **LLM:** Google Flan-T5 Base (Hugging Face Transformers)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Database:** ChromaDB (Local persistent storage)
* **Interface:** Gradio

---

## 📂 Project Structure

```text
rag-complaint-chatbot/
├── data/
│   ├── raw/                  # (Local only) Place 'complaints.csv' here
│   └── processed/            # Generated cleaned data
├── vector_store/             # ChromaDB vector index (Generated locally)
├── src/
│   ├── data_preprocessing.py # Task 1: Cleans & filters raw CSV data
│   ├── build_vector_store.py # Task 2: Chunking, Embedding & Indexing
│   └── rag_pipeline.py       # Task 3: The RAG Logic & RetrievalQA Chain
├── app.py                    # Task 4: Main Gradio Web Application
├── requirements.txt          # List of python dependencies
└── README.md                 # Project documentation

```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/zemenfes-afk/CrediTrust-Financial-Intelligent-Complaint-Analysis.git
cd CrediTrust-Financial-Intelligent-Complaint-Analysis

```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to avoid conflicts.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## 🏃‍♂️ How to Run

### Step 1: Prepare the Data

**Note:** The raw data file is too large for GitHub. You must download the financial complaints dataset (CSV) and place it in `data/raw/complaints.csv`.

Run the cleaning script to filter and normalize the data:

```bash
python src/data_preprocessing.py

```

### Step 2: Build the Knowledge Base (Vector Store)

This script converts text into vector embeddings and saves them to `vector_store/chroma_db`.
*(This process may take 5-10 minutes depending on your CPU).*

```bash
python src/build_vector_store.py

```

### Step 3: Launch the Chatbot

Start the Gradio web interface:

```bash
python app.py

```

After a few seconds, click the local URL displayed in the terminal (e.g., `http://127.0.0.1:7860`) to open the app in your browser.

---

## 🧪 Example Usage

**User Question:** > *"What are the common issues with Credit Cards?"*

**Bot Answer:**

> *"The common issues include login failures, systems being down, and problems making payments on time."*

**Evidence:**

> * **1. Credit Card:** *"always an issue with log in or multi system down..."*
> * **2. Credit Card:** *"trouble using the card... ridiculous fees..."*
> 
> 

---

## 🤝 Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes.
4. Push to the branch and open a Pull Request.

## 📜 License

This project is for educational purposes as part of the 10 Academy Challenge. Data is sourced from public financial complaint datasets.
