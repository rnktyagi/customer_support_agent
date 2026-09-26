import os 
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated , Literal , Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage , AIMessage
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score 

load_dotenv()

jev_client = TypeSafeClient(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api",
)

class State(TypedDict) :
    messages : Annotated[AnyMessage , add_messages]
    route: Optional[Literal["knowledge_base", "database_lookup", "human_escalate", "Irrelevant"]]
    urgency: Optional[Literal["low", "medium", "high"]]

def routing_decision_node(state : State) :
    messages = state.get("messages")

    current_query = messages[-1].content

    history = "\n".join(f"{m.type}: {m.content}"for m in messages[:-1])

    PROMPT=f"""
<role>
You are a decision model working in a customer support agent designed explicitly for routing between different supporting use cases based on the user query
</role>

<routing_decision>
According to the given user query, you have to decide between the given 4 different routing decisions :

1. Knowledge Base : If the user query is related to the company's knowledge base like policies, terms and conditions etc.
<example>What is your refund policy for cancelled orders?</example?
<example>How long does standard shipping usually take?</example>
<example>What payment methods do you accept</example>

2. Database Lookup : If the user query can be answered by providing the details to the user according to its user data from the company's database
<example>Where is my order right now?</example>
<example>How many orders have I placed in the last 3 months?</example>
<example>What is the current status of my refund?</example>

3. Human Escalation : If the user query involves frustration/dissatisfaction and requires urgent resolution by a human support.
<example>I was charged twice for the same order and need someone to look into it.</example>
<example>Someone accessed my account without my permission.</example>
<example>I want to file a complaint about how my issue was handled.</example>

4. Irrelvant : If the user query is completely irrelevant to your role kindly route it as irrelvant.
<example>Write me a Python program to sort an array.</example>
<example>What's the capital of France?</example>
<example>Tell me a joke about cats.</example>
"""
    try :
        response = jev_client.system_one(
            model="jev-1.13",
            state=f"{PROMPT}\n\nConversation History:\n{history}\n\nCurrent Query:\n{current_query}",
            questions={
                "route": Choice(
                    instructions="Based on the current query and conversation history, which routing decision applies?",
                    criteria={
                        "knowledge_base": "Answerable from company policies, terms, or general info.",
                        "database_lookup": "Needs the user's own account/order data.",
                        "human_escalate": "Frustration, dissatisfaction, or needs urgent human resolution.",
                        "Irrelevant": "Unrelated to customer support at all.",
                        },
                    ),
                "urgency": Score(
                    instructions="Determine the urgency of the customer's request based on frustration, potential harm, security concerns, or time sensitivity.",
                    criteria=[
                        "low: Normal customer request with no immediate risk or significant urgency.",
                        "medium: Important issue that should be resolved promptly, but there is no immediate serious risk.",
                        "high: Requires urgent attention because of security concerns, financial harm, serious dissatisfaction, or significant time sensitivity."
                    ]
                )     
            }
        )

        route=response.answers['route'].choice
        urgency=response.answers['urgency'].score

        return {"route" : route , "urgency" : urgency}
    except Exception as e :
        print(f"{e}")
        return {"route": None, "urgency": None}

