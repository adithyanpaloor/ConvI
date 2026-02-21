"""End-to-end integration test for the ConvI pipeline (text path + RAG + LLM)."""
from app.conversation_normalizer import normalize_from_text, turns_to_dialogue_string
from app.rag_engine import retriever
from app.llm_engine import run_llm_analysis

transcript = """Agent: Thank you for calling ConvI Bank. How can I help you today?
Customer: I am very angry. There is an unauthorized transaction of 5000 rupees on my account.
Agent: I am very sorry to hear that. Can I get your account number please?
Customer: Yes, it is 1234567890. This is fraud! Someone hacked my account.
Agent: I understand your concern. I will immediately block your card and file a dispute. You will get a refund within 7 business days.
Customer: Thank you. I hope this gets resolved soon.
Agent: Absolutely. Is there anything else I can help you with?
Customer: No that is all."""

# 1. Normalize
turns = normalize_from_text(transcript)
print(f"[1] Turns: {len(turns)}")

# 2. RAG
retriever.load()
query = " ".join(t.normalized_text_en for t in turns)[:500]
rag = retriever.retrieve(query)
print(f"[2] RAG chunks: {len(rag['rag_context_chunks'])}")

# 3. LLM
result = run_llm_analysis(turns, rag, domain="financial_banking")
basic = result["basic_conversational_analysis"]
print(f"[3] Summary      : {basic.conversation_summary}")
print(f"    Intent       : {basic.customer_intention}")
print(f"    Outcome      : {basic.call_outcome}")
print(f"    Risk         : {result['risk_score']} | Escalation: {result['escalation_level'].value}")
print(f"    Compliance   : {result['rag_based_analysis'].compliance_flags}")
print(f"    Fraud        : {result['rag_based_analysis'].fraud_indicators}")
print(f"    Agent score  : {result['agent_performance_analysis'].performance_score}")
print(f"    De-escalation: {result['agent_performance_analysis'].de_escalation_detected}")
print("\n[OK] All pipeline stages completed successfully.")
