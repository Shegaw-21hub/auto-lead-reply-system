import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, Header, HTTPException, status
from app.schemas.lead import RawLeadPayload, NormalizedLead
from app.workers.tasks import process_incoming_lead_task

app = FastAPI(title="Automatic Lead Reply System API")

@app.post("/api/v1/webhooks/leads", status_code=status.HTTP_202_ACCEPTED)
async def receive_lead_webhook(
    payload: RawLeadPayload,
    x_signature: str = Header(None)
):
    normalized_lead = NormalizedLead(
        lead_id=str(uuid.uuid4()),
        source=payload.source,
        full_name=f"{payload.first_name} {payload.last_name or ''}".strip(),
        email=payload.email,
        phone=payload.phone,
        company=payload.company,
        raw_message=payload.message,
        received_at=datetime.now(timezone.utc)
    )

    # Queue task asynchronously
    task = process_incoming_lead_task.delay(normalized_lead.model_dump(mode="json"))

    return {
        "status": "queued",
        "lead_id": normalized_lead.lead_id,
        "task_id": task.id
    }