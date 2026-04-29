"""
Base agent classes and configuration for OptiMat Alloys.

This module provides the foundation for creating agentic AI
assistants with different capabilities and personalities.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Callable
from abc import ABC, abstractmethod

from autogen_core.model_context import ChatCompletionContext


@dataclass
class AgentConfig:
    """
    Configuration for an agent.

    Attributes:
        name: Agent identifier
        system_message: Instructions defining agent behavior
        tools: List of tool functions the agent can call
        model_client: LLM client for the agent
        model_client_stream: Enable token streaming
        reflect_on_tool_use: Enable reflection after tool usage
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum response length
        model_context: Optional ChatCompletionContext to preserve conversation
            history across agent recreation (e.g., on system-message updates).
    """
    name: str
    system_message: str
    tools: List[Callable] = field(default_factory=list)
    model_client: Optional[Any] = None
    model_client_stream: bool = True
    reflect_on_tool_use: bool = True
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    model_context: Optional[ChatCompletionContext] = None


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
            Agent instance (e.g., AutoGen AssistantAgent)

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

        Args:
            tools: New list of tool functions

        Examples:
            >>> agent.update_tools([tool1, tool2])  # doctest: +SKIP
        """
        self.config.tools = tools
        if self._agent_instance is not None:
            # Preserve conversation history across recreation by re-using the
            # existing ChatCompletionContext via the public constructor path.
            self.config.model_context = self._agent_instance._model_context
            self._agent_instance = self.create_agent()

    def update_system_message(self, message: str) -> None:
        """
        Update the agent's system message.

        Args:
            message: New system message

        Examples:
            >>> agent.update_system_message("You are a helpful assistant")  # doctest: +SKIP
        """
        self.config.system_message = message
        if self._agent_instance is not None:
            # Preserve conversation history across recreation by re-using the
            # existing ChatCompletionContext via the public constructor path.
            self.config.model_context = self._agent_instance._model_context
            self._agent_instance = self.create_agent()
