"""
mcp_protocol.py
───────────────
Model Context Protocol (MCP) — Agent Communication Layer

MCP defines a standard message format for agent-to-agent communication.
Each message has:
  - role     : "user" | "agent" | "tool" | "system"
  - agent_id : which agent sent/receives this message
  - content  : the message text
  - tool_call: optional tool invocation request
  - tool_result: optional tool execution result
  - metadata : timestamps, tokens, scores, etc.

This enables:
  1. Structured agent communication logs
  2. Tool call / result tracking
  3. Multi-turn conversation context
  4. Agent handoff (one agent passing context to another)
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MCP Message
# ══════════════════════════════════════════════════════════════════════════════

class MCPMessage:
    """A single message in the MCP communication protocol."""

    def __init__(
        self,
        role: str,
        content: str,
        agent_id: str = "system",
        tool_call: Optional[dict] = None,
        tool_result: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.message_id: str = str(uuid.uuid4())[:8]
        self.role = role                    # user | agent | tool | system
        self.agent_id = agent_id            # AnalyticsAgent | PolicyAgent | ForecastAgent
        self.content = content
        self.tool_call = tool_call          # {"tool_name": ..., "params": {...}}
        self.tool_result = tool_result      # result string from tool execution
        self.metadata = metadata or {}
        self.timestamp: str = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        d = {
            "message_id": self.message_id,
            "role": self.role,
            "agent_id": self.agent_id,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.tool_call:
            d["tool_call"] = self.tool_call
        if self.tool_result is not None:
            d["tool_result"] = self.tool_result
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def __repr__(self) -> str:
        return f"MCPMessage(role={self.role}, agent={self.agent_id}, id={self.message_id})"


# ══════════════════════════════════════════════════════════════════════════════
# MCP Context — conversation thread
# ══════════════════════════════════════════════════════════════════════════════

class MCPContext:
    """
    Maintains the full conversation context for a multi-agent session.

    Tracks:
    - All messages (user, agent, tool calls, tool results)
    - Which agents were involved
    - Tool usage statistics
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())[:12]
        self.messages: list[MCPMessage] = []
        self.agents_involved: list[str] = []
        self.tools_called: list[str] = []
        self.created_at: str = datetime.now(timezone.utc).isoformat()

    def add(self, message: MCPMessage) -> None:
        """Add a message to the context."""
        self.messages.append(message)
        if message.agent_id not in self.agents_involved and message.agent_id != "system":
            self.agents_involved.append(message.agent_id)
        if message.tool_call:
            tool_name = message.tool_call.get("tool_name", "")
            if tool_name:
                self.tools_called.append(tool_name)

    def add_user_message(self, content: str) -> MCPMessage:
        msg = MCPMessage(role="user", content=content, agent_id="user")
        self.add(msg)
        return msg

    def add_agent_message(self, agent_id: str, content: str,
                          tool_call: Optional[dict] = None) -> MCPMessage:
        msg = MCPMessage(role="agent", content=content,
                         agent_id=agent_id, tool_call=tool_call)
        self.add(msg)
        return msg

    def add_tool_result(self, agent_id: str, tool_name: str, result: str) -> MCPMessage:
        msg = MCPMessage(
            role="tool",
            content=f"Tool '{tool_name}' returned result.",
            agent_id=agent_id,
            tool_result=result,
            metadata={"tool_name": tool_name},
        )
        self.add(msg)
        return msg

    def get_history_for_llm(self) -> list[dict]:
        """Return messages formatted for LLM chat history."""
        history = []
        for msg in self.messages:
            if msg.role == "user":
                history.append({"role": "user", "content": msg.content})
            elif msg.role == "agent":
                history.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool" and msg.tool_result:
                history.append({"role": "assistant",
                                 "content": f"[Tool Result] {msg.tool_result}"})
        return history

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "agents_involved": self.agents_involved,
            "tools_called": self.tools_called,
            "message_count": len(self.messages),
            "messages": [m.to_dict() for m in self.messages],
        }

    def summary(self) -> dict:
        """Return a compact summary without full message content."""
        return {
            "session_id": self.session_id,
            "agents_involved": self.agents_involved,
            "tools_called": self.tools_called,
            "message_count": len(self.messages),
        }

    @property
    def message_count(self) -> int:
        return len(self.messages)


# ══════════════════════════════════════════════════════════════════════════════
# MCP Orchestrator — routes messages between agents
# ══════════════════════════════════════════════════════════════════════════════

class MCPOrchestrator:
    """
    Routes user messages to the correct agent using MCP protocol.

    Implements the full MCP flow:
      1. Receive user message → create MCPContext
      2. Classify → select agent
      3. Agent selects tool → MCPMessage(tool_call)
      4. Tool executes → MCPMessage(tool_result)
      5. Agent generates response → MCPMessage(agent response)
      6. Return full context + final response
    """

    def __init__(self) -> None:
        from server.genai.agent_tools import TOOL_REGISTRY
        self.tool_registry = TOOL_REGISTRY

    def _select_tool_for_agent(self, agent_id: str, user_message: str) -> tuple[str, dict]:
        """Select the best tool for the agent based on the user message."""
        msg_lower = user_message.lower()

        if agent_id == "AnalyticsAgent":
            if any(w in msg_lower for w in ["top", "best", "highest", "leading"]):
                return "get_top_products", {}
            if any(w in msg_lower for w in ["category", "categories", "segment"]):
                return "get_category_breakdown", {}
            if any(w in msg_lower for w in ["region", "area", "location", "zone"]):
                return "get_region_breakdown", {}
            if any(w in msg_lower for w in ["monthly", "month", "trend", "timeline"]):
                return "get_monthly_trend", {}
            if any(w in msg_lower for w in ["product", "item", "sku", "detail"]):
                # Try to extract product ID
                params = {}
                for word in user_message.upper().split():
                    if word.startswith("PROD") and len(word) >= 6:
                        params["product_id"] = word
                        break
                return "get_product_detail", params
            return "get_sales_overview", {}

        elif agent_id == "PolicyAgent":
            return "search_policy", {"query": user_message}

        else:  # ForecastAgent
            if any(w in msg_lower for w in ["anomaly", "spike", "outlier", "unusual"]):
                return "get_anomaly_summary", {}
            if any(w in msg_lower for w in ["model", "algorithm", "feature", "training"]):
                return "get_model_info", {}
            return "get_demand_forecast", {
                "product_id": "PROD_001",
                "target_date": "2025-07-01",
                "price": 100.0,
                "discount": 0.0,
                "store_id": "STORE_A",
                "region": "North",
            }

    def process(self, user_message: str, session_id: Optional[str] = None) -> dict:
        """
        Full MCP pipeline:
        user message → classify → tool call → tool result → LLM response
        """
        from server.orchestration.query_handler import classify_query
        from server.genai.llm_client import call_llm
        from server.genai.analytics_agent import ANALYTICS_SYSTEM_PROMPT
        from server.genai.policy_agent import POLICY_SYSTEM_PROMPT
        from server.genai.forecast_agent import FORECAST_SYSTEM_PROMPT

        # Create MCP context
        ctx = MCPContext(session_id=session_id)

        # Step 1 — user message
        ctx.add_user_message(user_message)
        logger.info("MCP session %s started for: %s", ctx.session_id, user_message[:60])

        # Step 2 — classify → select agent
        agent_id = classify_query(user_message)

        # Step 3 — agent selects tool
        tool_name, tool_params = self._select_tool_for_agent(agent_id, user_message)
        ctx.add_agent_message(
            agent_id=agent_id,
            content=f"I will use the '{tool_name}' tool to answer your question.",
            tool_call={"tool_name": tool_name, "params": tool_params},
        )

        # Step 4 — execute tool
        tool_result = self.tool_registry.run(tool_name, tool_params)
        ctx.add_tool_result(agent_id, tool_name, tool_result)
        logger.info("MCP tool '%s' executed for agent '%s'.", tool_name, agent_id)

        # Step 5 — LLM generates final response
        system_prompts = {
            "AnalyticsAgent": ANALYTICS_SYSTEM_PROMPT,
            "PolicyAgent": POLICY_SYSTEM_PROMPT,
            "ForecastAgent": FORECAST_SYSTEM_PROMPT,
        }
        system_prompt = system_prompts.get(agent_id, ANALYTICS_SYSTEM_PROMPT)

        llm_response = call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            context=tool_result,
            max_tokens=450,
            temperature=0.2,
        )

        ctx.add_agent_message(agent_id=agent_id, content=llm_response)

        # Step 6 — return full MCP result
        return {
            "session_id": ctx.session_id,
            "message": user_message,
            "agent": agent_id,
            "tool_used": tool_name,
            "tool_result": tool_result,
            "response": llm_response,
            "mcp_context": ctx.summary(),
            "mcp_trace": ctx.to_dict(),
        }


# ── Singleton orchestrator ─────────────────────────────────────────────────────
_mcp_orchestrator: Optional[MCPOrchestrator] = None


def get_mcp_orchestrator() -> MCPOrchestrator:
    global _mcp_orchestrator
    if _mcp_orchestrator is None:
        _mcp_orchestrator = MCPOrchestrator()
    return _mcp_orchestrator
