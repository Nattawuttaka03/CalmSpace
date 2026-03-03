def build_prompt(user_input, context=""):
    return f"""
You are a helpful AI assistant.

### Context:
{context}

### User:
{user_input}

### Assistant:
"""