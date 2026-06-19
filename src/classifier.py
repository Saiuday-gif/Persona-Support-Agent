import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# Load environment variables from the .env file (e.g., GROQ_API_KEY)
load_dotenv()

# Define the desired JSON output structure using Pydantic
# This forces the LLM to return exactly these fields instead of plain text.
class PersonaClassification(BaseModel):
    persona: str = Field(
        description="Must be exactly one of: 'Technical Expert', 'Frustrated User', or 'Business Executive'"
    )
    justification: str = Field(
        description="Detailed reason for classifying the text into this specific persona"
    )

def classify_user_persona(user_message: str) -> dict:
    """
    Analyzes the user's message and classifies their communication persona.
    Returns a dictionary containing the detected persona and the justification.
    """
    
    # 1. Initialize the Groq LLM
    # We use a temperature of 0 to ensure deterministic, strict outputs (no creative hallucinations).
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0, 
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # 2. Bind the Pydantic schema to the LLM
    # This tells the model: "Do not return a normal paragraph. Return a JSON object matching this schema."
    structured_llm = llm.with_structured_output(PersonaClassification)
    
    # 3. Construct the prompt
    prompt = f"""
    Analyze the following user message and classify their communication persona.
    User Message: "{user_message}"
    """
    
    # 4. Generate the response and handle potential errors gracefully
    try:
        # Invoke the LLM with the prompt
        response = structured_llm.invoke(prompt)
        
        # The response is now a Pydantic object, so we access its attributes directly
        return {
            "persona": response.persona,
            "justification": response.justification
        }
    except Exception as e:
        # Fallback in case of API limits or formatting errors
        return {
            "persona": "Unknown",
            "justification": f"Error during classification: {str(e)}"
        }

# Standard Python idiom for executing testing logic when running this file directly
if __name__ == "__main__":
    # Let's test with a dummy message representing a frustrated user
    test_message = "I am extremely angry! I've been trying to log into my account for hours. Fix this right now!"
    
    print("Testing Classifier...")
    result = classify_user_persona(test_message)
    
    print(f"\nDetected Persona: {result['persona']}")
    print(f"Justification: {result['justification']}")