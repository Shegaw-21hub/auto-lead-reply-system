from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from app.agents.state import LeadAgentState
from app.config import settings

# Structured Output Schema
class LeadEvaluation(BaseModel):
    intent: Literal["Pricing Request", "Technical Question", "Partnership", "Spam"] = Field(
        description="Core intent of the lead inquiry"
    )
    budget_score: int = Field(description="Score 0-25 based on budget indications")
    authority_score: int = Field(description="Score 0-25 based on decision-maker role")
    need_score: int = Field(description="Score 0-25 based on pain point clarity")
    timeline_score: int = Field(description="Score 0-25 based on stated urgency")
    total_bant_score: int = Field(description="Sum of BANT scores (0-100)")
    reasoning: str = Field(description="Brief analysis of the lead quality")

# Node 1: AI Intent & BANT Evaluation Node
def classify_and_score_node(state: LeadAgentState) -> LeadAgentState:
    # Fallback if OPENAI_API_KEY is not configured yet
    if not settings.OPENAI_API_KEY:
        return {
            **state,
            "intent": "Pricing Request",
            "bant_score": 85,
            "needs_human_review": False
        }

    llm = ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY, temperature=0)
    structured_llm = llm.with_structured_output(LeadEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI sales intelligence engine. Analyze incoming lead inquiries and output structured BANT scores."),
        ("user", "Lead Name: {full_name}\nCompany: {company}\nMessage: {raw_message}")
    ])

    chain = prompt | structured_llm
    
    evaluation: LeadEvaluation = chain.invoke({
        "full_name": state.get("full_name", ""),
        "company": state.get("company", "Unknown"),
        "raw_message": state.get("raw_message", "")
    })

    return {
        **state,
        "intent": evaluation.intent,
        "bant_score": evaluation.total_bant_score,
        "needs_human_review": evaluation.total_bant_score < 70 or evaluation.intent == "Spam"
    }

# Build State Graph
builder = StateGraph(LeadAgentState)
builder.add_node("classify_and_score", classify_and_score_node)
builder.set_entry_point("classify_and_score")
builder.add_edge("classify_and_score", END)

lead_agent_graph = builder.compile()