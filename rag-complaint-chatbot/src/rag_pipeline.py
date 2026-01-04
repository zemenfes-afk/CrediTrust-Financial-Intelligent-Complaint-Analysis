import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# Try classic import first, fallback to standard
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class RAGApplication:
    def __init__(self, vector_store_path="../vector_store/chroma_db"):
        print("Initializing RAG Pipeline...")

        # 1. Load Vector Store
        print("Loading Vector Store...")
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if not os.path.exists(vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {vector_store_path}. Run Task 2 first.")

        self.vector_db = Chroma(
            persist_directory=vector_store_path,
            embedding_function=self.embedding_model
        )

        # 2. Setup Retriever
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

        # 3. Setup LLM (Using 'base' version to save RAM)
        # --- CHANGED MODEL HERE ---
        model_id = "google/flan-t5-base"
        print(f"Loading LLM ({model_id})...")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.1,
            repetition_penalty=1.1
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

        # 4. Prompt Template
        template = """You are a financial analyst assistant. 
        Use the context below to answer the question.

        Context: {context}
        Question: {question}
        Answer:"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # 5. Build Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        print("✅ RAG Pipeline is ready!")

    def run(self, query):
        result = self.qa_chain.invoke({"query": query})
        return {
            "question": query,
            "answer": result['result'],
            "source_documents": result['source_documents']
        }


if __name__ == "__main__":
    app = RAGApplication()
    print(app.run("What are the common issues with Credit Cards?"))