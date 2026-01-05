import gradio as gr
import sys
import os

# 1. Setup path to import our own code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from rag_pipeline import RAGApplication

# 2. Initialize the Chatbot
print("⏳ Loading RAG System... Please wait.")
rag_system = RAGApplication(vector_store_path="vector_store/chroma_db")


def chat_logic(message, history):
    """
    This function connects the user's input (message) to our RAG backend.
    """
    if not message:
        return ""

    # Run the RAG pipeline
    response_data = rag_system.run(message)

    answer = response_data['answer']
    sources = response_data['source_documents']

    # Format the sources nicely for the UI
    formatted_sources = "\n\n---\n**🔎 Evidence from Complaints:**\n"
    for i, doc in enumerate(sources, 1):
        product = doc.metadata.get('product', 'Unknown Product')
        snippet = doc.page_content[:150].replace("\n", " ") + "..."
        formatted_sources += f"**{i}. {product}:** _{snippet}_\n\n"

    # Combine the main Answer + The Evidence
    full_response = f"**🤖 Analysis:**\n{answer}\n{formatted_sources}"
    return full_response


# 3. Build the User Interface
# We remove 'theme' here to fix the warning and pass it in launch() if needed,
# or just rely on default. For now, we keep it simple.
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏦 CrediTrust Financial Chatbot
        **Ask detailed questions about consumer complaints.** _Example: "What are the common issues with Student Loans?"_
        """
    )

    chat_interface = gr.ChatInterface(
        fn=chat_logic,
        chatbot=gr.Chatbot(height=450),
        textbox=gr.Textbox(placeholder="Type your question here...", container=False, scale=7),
        title=None,
        description=None,
        # REMOVED THE INVALID 'theme="soft"' LINE HERE
        examples=[
            "What are the common issues with Credit Cards?",
            "Why do people complain about Mortgages?",
            "Are there hidden fees in Checking Accounts?",
            "How long do money transfers take?"
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    print("🚀 Starting the User Interface...")
    # Fix the warning by passing allowed_paths if needed, or just launch
    demo.launch()