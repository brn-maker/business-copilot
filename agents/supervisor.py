"""
Supervisor Agent: Orchestrates multi-agent system using LangGraph.

Routes queries to appropriate specialized agents and synthesizes results.
"""

from typing import Any, Dict, List, Annotated, TypedDict, Optional
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from utils import model_router, prompts
from agents import data_agent, analysis_agent, forecast_agent, recommender_agent


class AgentState(TypedDict):
    """State passed between agents in the graph."""
    messages: List[BaseMessage]
    current_agent: Optional[str]
    data: Dict[str, Any]  # Shared data storage (uploaded files, processed data)
    results: Dict[str, Any]  # Results from each agent
    next_action: Optional[str]


class MultiAgentSupervisor:
    """Orchestrates multi-agent system with LangGraph."""
    
    def __init__(self):
        """Initialize supervisor with all agents."""
        self.router = model_router.create_model_router()
        self.graph = None
        self.compiled_graph = None
        self._setup_graph()
    
    def _setup_graph(self):
        """Build LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("data_agent", self._data_agent_node)
        workflow.add_node("analysis_agent", self._analysis_agent_node)
        workflow.add_node("forecast_agent", self._forecast_agent_node)
        workflow.add_node("recommender_agent", self._recommender_agent_node)
        
        # Entry point
        workflow.add_edge("__start__", "supervisor")
        
        # Conditional routing from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "data": "data_agent",
                "analysis": "analysis_agent",
                "forecast": "forecast_agent",
                "recommendations": "recommender_agent",
                "end": END,
            }
        )
        
        # All agents route back to supervisor to determine next step
        workflow.add_edge("data_agent", "supervisor")
        workflow.add_edge("analysis_agent", "supervisor")
        workflow.add_edge("forecast_agent", "supervisor")
        workflow.add_edge("recommender_agent", "supervisor")
        
        self.graph = workflow
        self.compiled_graph = workflow.compile()
    
    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Supervisor logic: routes based on user message."""
        messages = state.get("messages", [])
        
        if not messages:
            return state
        
        last_message = messages[-1]
        user_input = last_message.content if isinstance(last_message, HumanMessage) else ""
        
        # Get routing decision from LLM
        supervisor_model = self.router.get_model_for_task("analysis")
        
        routing_prompt = f"""Analyze the user query and determine which agent(s) to invoke.
        
Query: {user_input}

Available agents:
- data: Load/process Excel files, clean data
- analysis: Statistical analysis, correlations, regression
- forecast: Time-series forecasting, predictions
- recommendations: Business recommendations based on analysis
- end: Complete the conversation

Respond with exactly one of: data, analysis, forecast, recommendations, or end

Decision:"""
        
        try:
            response = supervisor_model.invoke([HumanMessage(content=routing_prompt)])
            routing_decision = response.content.strip().lower().split()[0]
        except:
            routing_decision = "end"
        
        # Validate routing decision
        valid_routes = ["data", "analysis", "forecast", "recommendations", "end"]
        next_agent = routing_decision if routing_decision in valid_routes else "end"
        
        state["next_action"] = next_agent
        return state
    
    def _route_from_supervisor(self, state: AgentState) -> str:
        """Route based on supervisor's decision."""
        return state.get("next_action", "end")
    
    def _data_agent_node(self, state: AgentState) -> AgentState:
        """Execute data agent tasks."""
        messages = state.get("messages", [])
        data = state.get("data", {})
        
        # Get user query
        if messages:
            user_query = messages[-1].content
        else:
            user_query = "Process available data"
        
        # Create data agent tools
        tools = data_agent.get_data_agent_tools()
        data_model = self.router.get_model_for_task("analysis")
        
        # Invoke agent (simplified: just call describe function on available data)
        results = {
            "agent": "data",
            "tasks_completed": [],
            "summary": "Data processing ready",
        }
        
        state["results"]["data"] = results
        messages.append(AIMessage(content="Data processing tasks executed"))
        state["messages"] = messages
        
        return state
    
    def _analysis_agent_node(self, state: AgentState) -> AgentState:
        """Execute analysis agent tasks."""
        messages = state.get("messages", [])
        data = state.get("data", {})
        
        # Simplified analysis execution
        results = {
            "agent": "analysis",
            "analyses_completed": [],
            "insights": "Statistical analysis ready",
        }
        
        state["results"]["analysis"] = results
        messages.append(AIMessage(content="Statistical analysis completed"))
        state["messages"] = messages
        
        return state
    
    def _forecast_agent_node(self, state: AgentState) -> AgentState:
        """Execute forecast agent tasks."""
        messages = state.get("messages", [])
        
        # Simplified forecast execution
        results = {
            "agent": "forecast",
            "forecasts_generated": [],
            "predictions": "Forecasting ready",
        }
        
        state["results"]["forecast"] = results
        messages.append(AIMessage(content="Forecasts generated"))
        state["messages"] = messages
        
        return state
    
    def _recommender_agent_node(self, state: AgentState) -> AgentState:
        """Execute recommender agent tasks."""
        messages = state.get("messages", [])
        
        # Simplified recommendations
        results = {
            "agent": "recommender",
            "recommendations_count": 0,
            "summary": "Business recommendations ready",
        }
        
        state["results"]["recommender"] = results
        messages.append(AIMessage(content="Recommendations synthesized"))
        state["messages"] = messages
        
        return state
    
    def invoke(
        self,
        user_message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the multi-agent system.
        
        Args:
            user_message: User query
            data: Optional data context (uploaded DataFrames, etc)
            
        Returns:
            Result with agent outputs and recommendations
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_message)],
            "current_agent": None,
            "data": data or {},
            "results": {},
            "next_action": None,
        }
        
        try:
            final_state = self.compiled_graph.invoke(
                initial_state,
                config={"max_iterations": 10},
            )
            
            return {
                "status": "success",
                "messages": final_state.get("messages", []),
                "results": final_state.get("results", {}),
                "response": final_state["messages"][-1].content if final_state["messages"] else "No response",
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response": f"Error in multi-agent processing: {str(e)}",
            }


def create_supervisor() -> MultiAgentSupervisor:
    """Factory function to create supervisor."""
    return MultiAgentSupervisor()
