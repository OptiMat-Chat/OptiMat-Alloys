"""
Scientist agent implementation for atomistic simulations.

This module provides the Scientist agent that assists users with
materials science simulations and analysis.
"""

from typing import Any, List, Callable, Optional
from autogen_agentchat.agents import AssistantAgent
from .base import BaseAgent, AgentConfig


class ScientistAgent(BaseAgent):
    """
    Scientist agent specialized in atomistic simulations.

    Provides access to tools for generating and analyzing atomic structures,
    computing properties, and visualizing results.
    """

    def create_agent(self) -> AssistantAgent:
        """
        Create an AutoGen AssistantAgent configured as a Scientist.

        Returns:
            Configured AssistantAgent instance

        Examples:
            >>> config = AgentConfig(
            ...     name="Scientist",
            ...     system_message="You are a scientist",
            ...     model_client=client
            ... )
            >>> agent = ScientistAgent(config)  # doctest: +SKIP
            >>> instance = agent.create_agent()  # doctest: +SKIP
        """
        kwargs = dict(
            name=self.config.name,
            model_client=self.config.model_client,
            system_message=self.config.system_message,
            tools=self.config.tools,
            model_client_stream=self.config.model_client_stream,
            reflect_on_tool_use=self.config.reflect_on_tool_use,
        )
        if self.config.model_context is not None:
            kwargs["model_context"] = self.config.model_context
        return AssistantAgent(**kwargs)


def create_scientist_agent(
    model_client: Any,
    tools: List[Callable],
    name: str = "Scientist",
    system_message: Optional[str] = None,
    model_client_stream: bool = True,
    reflect_on_tool_use: bool = True,
) -> ScientistAgent:
    """
    Convenience function to create a Scientist agent.

    Args:
        model_client: LLM client for the agent
        tools: List of tool functions
        name: Agent name
        system_message: Custom system message (uses default if None)
        model_client_stream: Enable streaming
        reflect_on_tool_use: Enable reflection

    Returns:
        Configured ScientistAgent

    Examples:
        >>> agent = create_scientist_agent(  # doctest: +SKIP
        ...     model_client=client,
        ...     tools=[generate_alloy_supercell]
        ... )
    """
    # Import the canonical system message from the factory to maintain a single source of truth
    from .factory import AgentFactory
    default_message = AgentFactory.DEFAULT_MESSAGES["scientist"]

    config = AgentConfig(
        name=name,
        system_message=system_message or default_message,
        tools=tools,
        model_client=model_client,
        model_client_stream=model_client_stream,
        reflect_on_tool_use=reflect_on_tool_use,
    )

    return ScientistAgent(config)
