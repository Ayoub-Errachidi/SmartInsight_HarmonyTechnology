# views/stats.py
from django.http import StreamingHttpResponse
from django.views.decorators.http import require_GET
from Utils.ollama_ai import stream_ai_explanation
import logging

logger = logging.getLogger(__name__)

@require_GET
def stream_explanation_stats(request):
    prompt = request.GET.get("prompt", "")
    if not prompt:
        return StreamingHttpResponse("[Erreur] Aucun prompt fourni.", content_type="text/plain")

    def event_stream():
        try:
            # Stream de l'explication IA en continu
            for chunk in stream_ai_explanation(prompt):
                yield f"data: {chunk}\n\n"  # Ajout du préfixe 'data:' pour respecter le format Server-Sent Events
        except Exception as e:
            logger.error(f"Erreur lors du streaming de l'explication : {str(e)}")  # Log de l'erreur
            yield f"data: [Erreur de serveur] {str(e)}\n\n"

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
