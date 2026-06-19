def check_for_escalation(user_message: str, persona: str) -> bool:
    """
    Checks if the conversation needs to be handed over to a human agent.
    """
    # Logic: If user is "Frustrated" and uses strong keywords, escalate!
    escalation_keywords = ["human", "person", "representative", "manager", "support team"]
    
    if persona == "Frustrated User":
        if any(word in user_message.lower() for word in escalation_keywords):
            return True
    return False

def get_escalation_message():
    return "I understand your frustration. Connecting you to a human support representative immediately..."