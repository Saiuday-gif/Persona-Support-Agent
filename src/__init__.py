# Exposing core functions to the src package
from .classifier import classify_user_persona
from .generator import generate_support_response
from .escalator import check_for_escalation, get_escalation_message

# This makes importing much cleaner in app.py