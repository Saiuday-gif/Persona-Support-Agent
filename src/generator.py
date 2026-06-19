import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

def generate_support_response(user_message: str, detected_persona: str) -> str:
    """
    Retrieves relevant context from the Chroma vector database and generates 
    a personalized response using the Groq LLM based on the user's persona.
    """
    
    # 1. Initialize the same embedding model used during the RAG pipeline setup
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to the existing local Chroma vector database
    try:
        vector_db = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=embeddings
        )
    except Exception as e:
        return f"System Error: Could not connect to the database. Details: {e}"

    # 3. Retrieve the top 3 most relevant document chunks based on the user's message
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(user_message)
    
    # Combine the text from the retrieved documents into a single context string
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 4. Initialize the Groq LLM
    # We use a slightly higher temperature (0.3) here to allow the AI to adjust its tone naturally
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3, 
        api_key=os.getenv("GROQ_API_KEY")
    )

    # 5. Define the Prompt Template instructing the AI how to behave
    prompt_template = """
    You are an expert AI Customer Support Agent. 
    Analyze the provided Context and answer the User Question accurately.
    
    IMPORTANT RULES:
    - Adapt your tone and vocabulary to strictly match this user persona: {persona}
    - If "Frustrated User": Be highly empathetic, apologetic, and clear. Avoid sounding robotic.
    - If "Technical Expert": Use technical jargon, be precise, and get straight to the point. Do not over-explain basics.
    - If "Business Executive": Be highly professional, concise, and focus on business impact (e.g., SLAs, resolutions).
    - Base your answer ONLY on the provided Context. If the context does not contain the answer, politely state that you do not have that information. Do not guess.

    Context:
    {context}

    User Question:
    {question}

    Answer:
    """

    # 6. Format the prompt with the dynamic variables
    prompt = PromptTemplate(
        input_variables=["persona", "context", "question"],
        template=prompt_template
    )

    formatted_prompt = prompt.format(
        persona=detected_persona, 
        context=context, 
        question=user_message
    )
    
    # 7. Generate and return the final response
    try:
        response = llm.invoke(formatted_prompt)
        return response.content
    except Exception as e:
        return f"Sorry, I encountered an error while generating the response. Error: {str(e)}"

# Standard testing block to verify the generator logic independently
if __name__ == "__main__":
    # Dummy inputs for testing
    test_query = "What is the billing policy?"
    test_persona = "Business Executive"
    
    print("Searching database and generating AI response...\n")
    answer = generate_support_response(test_query, test_persona)
    
    print("--- AI Agent Response ---")
    print(answer)