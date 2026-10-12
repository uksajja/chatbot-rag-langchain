
import yaml
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

def load_prompt_template(system_prompt_path: str | Path) -> ChatPromptTemplate:
    """
    Load a prompt template from a file.

    Args:
        system_prompt_path (str | Path): The path to the system prompt file.

    Returns:
        ChatPromptTemplate: The loaded prompt template.
    
    my_prompt.txt:
    System: You are a helpful assistant.
    User: {question}

    Example:
        system_prompt_path = "config/system_prompts.yaml"
        prompt_template = load_prompt_template(system_prompt_path)
        prompt = prompt_template.invoke({"question": "Hello, world!", "context": ""})
    
    """
    with open(system_prompt_path, encoding="utf-8") as file:
        prompt_data = yaml.safe_load(file)

    system_prompt = prompt_data["system_prompts"]["troubleshooting"]

    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}\n\nReference context:\n{context}"),
    ])
    
    return template
