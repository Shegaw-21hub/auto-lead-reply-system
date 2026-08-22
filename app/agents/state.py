from typing import TypedDict, Optional, List

class LeadAgentState(TypedDict):
    full_name: str
    email: str
    company: Optional[str]
    raw_message: str
    
    # Processed states
    intent: Optional[str]
    bant_score: Optional[int]
    retrieved_context: Optional[List[str]]
    generated_reply: Optional[str]
    needs_human_review: Optional[bool]