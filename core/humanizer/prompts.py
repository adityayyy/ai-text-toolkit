"""System prompts - single source of truth for all prompts."""

from config.constants import HUMANIZE_SYSTEM_PROMPT

# Re-export for backward compatibility
__all__ = ["HUMANIZE_SYSTEM_PROMPT"]

# Additional prompts can be added here as needed
DETECTION_PROMPT_TEMPLATE = """Analyze the following text and assess the likelihood it was written by 
an AI language model versus a human. Respond in this exact format:

LIKELIHOOD: [0-100]
REASONING: [one sentence]

Text: {text}"""