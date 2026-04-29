"""
Base tool class for agent tools.

This module provides the foundation for all Chainlit-aware agent tools,
handling common patterns like task list management and async execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import chainlit as cl


class BaseTool(ABC):
    """
    Base class for all agent tools.

    Tools wrap core business logic with Chainlit-specific UI handling
    (task lists, progress updates, visualization).

    Examples:
        >>> class MyTool(BaseTool):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_tool"
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Does something useful"
        ...
        ...     async def execute(self, param: str) -> Dict:
        ...         return {"result": param.upper()}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (used by agent to call tool)"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description (shown to agent)"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute tool logic.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Dictionary with tool results
        """
        pass

    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Chainlit-compatible wrapper for execute().

        This allows the tool to be used directly as a Chainlit tool function.
        """
        return await self.execute(**kwargs)

    async def create_task_list(self, tasks: list[str]) -> cl.TaskList:
        """
        Create a Chainlit task list for tracking progress.

        Args:
            tasks: List of task descriptions

        Returns:
            TaskList object with tasks in READY state
        """
        task_list = cl.TaskList()
        task_list.status = "Running..."

        for title in tasks:
            task = cl.Task(title=title, status=cl.TaskStatus.READY)
            await task_list.add_task(task)

        await task_list.send()
        return task_list

    async def update_task_status(
        self,
        task_list: cl.TaskList,
        task_index: int,
        status: cl.TaskStatus
    ):
        """
        Update status of a specific task.

        Args:
            task_list: The task list to update
            task_index: Index of task to update (0-based)
            status: New status (READY, RUNNING, DONE, FAILED)
        """
        task_list.tasks[task_index].status = status
        await task_list.send()

    async def complete_task_list(self, task_list: cl.TaskList, success: bool = True):
        """
        Mark task list as complete.

        Args:
            task_list: The task list to complete
            success: Whether all tasks succeeded
        """
        task_list.status = "Done!" if success else "Failed"
        await task_list.send()


class ToolRegistry:
    """
    Registry for managing available tools.

    Provides centralized tool management for agents.

    Examples:
        >>> registry = ToolRegistry()
        >>> registry.register(MyTool())
        >>> tool = registry.get("my_tool")
        >>> all_tools = registry.get_all()
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool by its name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def get_all(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_tool_functions(self) -> list:
        """
        Get all tools as callable functions for AutoGen.

        Returns:
            List of tool functions ready for AssistantAgent
        """
        return [tool.__call__ for tool in self._tools.values()]

    def list_names(self) -> list[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())
