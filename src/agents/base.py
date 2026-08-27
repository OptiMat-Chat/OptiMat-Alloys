"""
Base agent classes and configuration for OptiMat Alloys.

This module provides the foundation for creating agentic AI
assistants with different capabilities and personalities.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Callable
from abc import ABC, abstractmethod


@dataclass
class AgentConfig:
    """
    Configuration for an agent.

    Attributes:
        name: Agent identifier
        system_message: Instructions defining agent behavior. NOTE: the
            Agent Framework agent is constructed WITHOUT instructions; this
            text is delivered per run (optionally prefixed with session
            context) so it stays the sole system message while conversation
            history lives in the AgentSession.
        tools: List of tool functions the agent can call
        model_client: Agent Framework chat client for the agent
        temperature: Sampling temperature (0-2), applied via the agent's
            default_options
        max_tokens: Maximum response length (reserved; not currently
            applied — parity with the AutoGen-era behavior)
    """
    name: str
    system_message: str
    tools: List[Callable] = field(default_factory=list)
    model_client: Optional[Any] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Agents are AI assistants with specific capabilities and behaviors
    defined by their system messages and available tools.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize agent with configuration.

        Args:
            config: Agent configuration object
        """
        self.config = config
        self._agent_instance: Optional[Any] = None

    @abstractmethod
    def create_agent(self) -> Any:
        """
        Create the underlying agent instance.

        Returns:
            Agent instance (agent_framework.Agent)

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement create_agent()")

    def get_agent(self) -> Any:
        """
        Get or create the agent instance.

        Returns:
            Agent instance, creating it if necessary

        Examples:
            >>> config = AgentConfig(name="test", system_message="test")
            >>> agent = MyAgent(config)  # doctest: +SKIP
            >>> instance = agent.get_agent()  # doctest: +SKIP
        """
        if self._agent_instance is None:
            self._agent_instance = self.create_agent()
        return self._agent_instance

    def update_tools(self, tools: List[Callable]) -> None:
        """
        Update the tools available to this agent.

        Conversation history is unaffected: it lives in the AgentSession,
        not the agent instance, so recreation is safe.

        Args:
            tools: New list of tool functions

        Examples:
            >>> agent.update_tools([tool1, tool2])  # doctest: +SKIP
        """
        self.config.tools = tools
        if self._agent_instance is not None:
            self._agent_instance = self.create_agent()

    def update_system_message(self, message: str) -> None:
        """
        Update the agent's system message.

        The agent instance does not hold instructions (they are delivered
        per run from this config), so no recreation is needed.

        Args:
            message: New system message

        Examples:
            >>> agent.update_system_message("You are a helpful assistant")  # doctest: +SKIP
        """
        self.config.system_message = message
