# Utils/ollama_ai.py
import ollama
import logging

# Initialiser le logger
logger = logging.getLogger(__name__)

def stream_ai_explanation(prompt):
    
    try:
        response = ollama.chat(
            model='llama3',       #depseeek-r1
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in response:
            content = chunk.get('message', {}).get('content')
            if content:
                yield content
    except ollama.OllamaError as oe:
        error_msg = f"[Erreur Ollama] {str(oe)}"
        logger.error(error_msg)
        yield error_msg
    except Exception as e:
        error_msg = f"[Erreur inconnue] {str(e)}"
        logger.exception(error_msg)
        yield error_msg
