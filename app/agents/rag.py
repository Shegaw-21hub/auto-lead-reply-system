from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from app.agents.state import LeadAgentState
from app.agents.rag import query_knowledge_base
from app.config import settings

class LeadEvaluation(BaseModel):
    intent: Literal["Pricing Request", "Technical Question", "Partnership", "Spam"] = Field(
        description="Core intent of the lead inquiry"
    )
    total_bant_score: int = Field(description="Sum of BANT scores from 0-100")

# Node 1: Intent & BANT Scoring Node
def classify_and_score_node(state: LeadAgentState) -> LeadAgentState:
    message_text = state.get("raw_message", "")
    
    if not settings.OPENAI_API_KEY:
        intent = "Pricing Request" if "pricing" in message_text.lower() or "quote" in message_text.lower() else "Technical Question"
        score = 85 if intent == "Pricing Request" else 65
    else:
        llm = ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY, temperature=0)
        structured_llm = llm.with_structured_output(LeadEvaluation)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI sales intelligence engine. Evaluate lead inquiries using BANT criteria."),
            ("user", "Lead Name: {full_name}\nCompany: {company}\nMessage: {raw_message}")
        ])
        
        eval_result = (prompt | structured_llm).invoke({
            "full_name": state.get("full_name", ""),
            "company": state.get("company", "Unknown"),
            "raw_message": message_text
        })
        intent = eval_result.intent
        score = eval_result.total_bant_score

    return {
        **state,
        "intent": intent,
        "bant_score": score,
        "needs_human_review": score < 70 or intent == "Spam"
    }

# Node 2: Pinecone RAG Vector Context Node
def rag_retrieval_node(state: LeadAgentState) -> LeadAgentState:
    raw_message = state.get("raw_message", "")
    retrieved_docs = query_knowledge_base(query_text=raw_message, top_k=2)
    
    return {
        **state,
        "retrieved_context": retrieved_docs
    }

# Node 3: Personalized Email Generator Node
def response_generator_node(state: LeadAgentState) -> LeadAgentState:
    context_str = "\n".join(state.get("retrieved_context", []))
    
    if not settings.OPENAI_API_KEY:
        reply = (
            f"Hi {state.get('full_name')},\n\n"
            f"Thank you for contacting us regarding {state.get('company')}. "
            f"Based on your request, here are our terms:\n{context_str}\n\n"
            f"Would you like to schedule a 15-minute call? Book here: https://calendly.com/sales/15min"
        )
    else:
        llm = ChatOpenAI(model="gpt-4o", api_key=settings.OPENAI_API_KEY, temperature=0.3)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI Sales Representative. Craft a personalized, professional email response based strictly on the provided knowledge context."),
            ("user", "Lead Name: {full_name}\nCompany: {company}\nInquiry: {raw_message}\nKnowledge Context:\n{context}")
        ])
        
        response = (prompt | llm).invoke({
            "full_name": state.get("full_name", ""),
            "company": state.get("company", "Unknown"),
            "raw_message": state.get("raw_message", ""),
            "context": context_str
        })
        reply = response.content

    return {
        **state,
        "generated_reply": reply
    }

# Build LangGraph State Graph
builder = StateGraph(LeadAgentState)

builder.add_node("classify_and_score", classify_and_score_node)
builder.add_node("rag_retrieval", rag_retrieval_node)
builder.add_node("response_generator", response_generator_node)

builder.set_entry_point("classify_and_score")
builder.add_edge("classify_and_score", "rag_retrieval")
builder.add_edge("rag_retrieval", "response_generator")
builder.add_edge("response_generator", END)

lead_agent_graph = builder.compile()