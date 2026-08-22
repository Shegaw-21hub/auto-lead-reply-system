import logging
from app.workers.celery_app import celery_app
from app.agents.graph import lead_agent_graph

logger = logging.getLogger(__name__)

@celery_app.task(name="tasks.process_incoming_lead", bind=True, max_retries=3)
def process_incoming_lead(self, lead_data: dict):
    """
    Background worker task to process normalized incoming leads.
    """
    logger.info(f"Starting processing for lead: {lead_data.get('email')}")
    
    try:
        # Run state graph orchestration
        result = lead_agent_graph.invoke(lead_data)
        
        logger.info(
            f"Successfully processed lead {lead_data.get('email')}. "
            f"BANT Score: {result.get('bant_score')}, Human Review: {result.get('needs_human_review')}"
        )
        return result

    except Exception as exc:
        logger.error(f"Error processing lead {lead_data.get('email')}: {exc}")
        # Retry with exponential backoff on transient errors
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)