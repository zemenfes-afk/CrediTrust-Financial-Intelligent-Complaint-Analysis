import pandas as pd
from rag_pipeline import RAGApplication


def evaluate():
    # 1. Initialize the System
    rag = RAGApplication()

    # 2. Define Test Questions (Based on your dataset)
    test_questions = [
        "What are the main complaints regarding Credit Cards?",
        "Why are customers unhappy with their Savings Accounts?",
        "Are there issues with Money Transfers being delayed?",
        "How do customers describe their experience with Personal Loans?",
        "What is the most common issue with late fees?"
    ]

    results = []

    print("\n--- Starting Evaluation ---")
    for i, q in enumerate(test_questions):
        print(f"Testing Q{i + 1}: {q}")
        response = rag.ask(q)

        # Get the first source's product category for verification
        top_source = response['source_documents'][0]['product'] if response['source_documents'] else "No Source"

        results.append({
            "Question": q,
            "Generated Answer": response['answer'],
            "Top Source Category": top_source
        })

    # 3. Create DataFrame and Save
    df_results = pd.DataFrame(results)

    # Save to CSV for your report
    output_path = "../reports/rag_evaluation_results.csv"
    df_results.to_csv(output_path, index=False)

    print("\n--- Evaluation Complete ---")
    print(df_results)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    evaluate()