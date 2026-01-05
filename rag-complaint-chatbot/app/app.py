import gradio as gr
import sys
import os

# 1. Setup path to import our own code
# This tells Python to look inside the 'src' folder for our scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from rag_pipeline import RAGApplication

# 2. Initialize the Chatbot
print("⏳ Loading RAG System... Please wait.")
# We point to the vector store we built in Task 2
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
        # Show first 150 characters of the complaint
        snippet = doc.page_content[:150].replace("\n", " ") + "..."

        formatted_sources += f"**{i}. {product}:** _{snippet}_\n\n"

    # Combine the main Answer + The Evidence
    full_response = f"**🤖 Analysis:**\n{answer}\n{formatted_sources}"
    return full_response


# 3. Build the User Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏦 CrediTrust Financial Chatbot
        **Ask detailed questions about consumer complaints.** _Example: "What are the common issues with Student Loans?"_
        """
    )

    with gr.Row():
        # Chat Interface
        chat_interface = gr.ChatInterface(
            fn=chat_logic,
            chatbot=gr.Chatbot(height=450),
            textbox=gr.Textbox(placeholder="Type your question here...", container=False, scale=7),
            title=None,
            description=None,
            theme="soft",
            examples=[
                "What are the common issues with Credit Cards?",
                "Why do people complain about Mortgages?",
                "Are there hidden fees in Checking Accounts?",
                "How long do money transfers take?"
            ],
            cache_examples=False,  # Disable caching to save time
        )

if __name__ == "__main__":
    print("🚀 Starting the User Interface...")
    demo.launch()