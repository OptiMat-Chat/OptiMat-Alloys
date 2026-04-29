"""
Agent factory for creating different types of agents.

This module provides a centralized way to create and configure agents
with consistent patterns and defaults.
"""

from typing import List, Callable, Optional, Any, Dict
from .base import AgentConfig, BaseAgent
from .scientist import ScientistAgent


class AgentFactory:
    """
    Factory for creating agents with different configurations.

    Centralizes agent creation logic and provides defaults for
    common agent types.
    """

    # Default system messages for different agent types
    DEFAULT_MESSAGES: Dict[str, str] = {
        "scientist": (
            "You are an expert materials science AI assistant with access to atomistic simulation tools.\n\n"
            "TOOL EXECUTION (CRITICAL):\n"
            "ALWAYS call tools immediately when the user's intent is clear. "
            "Do NOT describe what you will do — just call the tool. Never ask for confirmation before calling a tool. "
            "Exception: When the user asks about PROPERTIES of an alloy (or asks to COMPARE alloys), "
            "call search_database FIRST for EACH alloy — "
            "do NOT call generate_alloy_supercell, calculate_elastic_properties, or compute_anharmonic_properties "
            "until you have checked whether the data already exists.\n\n"
            "TOOL INTERPRETATION:\n"
            "Analyze tool results in materials science context. Interpret: formation energy (negative=stable), "
            "structural analysis (PTM fractions), density, convergence. Never echo raw output.\n\n"
            "TOOL SELECTION:\n"
            "• ANY question about properties of an alloy → ALWAYS call search_database FIRST.\n"
            "• COMPARE/CONTRAST two or more alloys → call search_database for EACH alloy FIRST, then present results side-by-side. Pass calculator_name if specified. Only compute if data is missing.\n"
            "• RANKING/SUPERLATIVE queries ('Which alloy is stiffest?', 'Best bulk modulus?', 'Most stable FCC?') → search_database ONLY. Never compute to answer ranking questions.\n"
            "• 'Do we have / What do we know about X?' → search_database ONLY.\n"
            "  - If search results show has_elastic_properties=True or has_qha_data=True, present those values.\n"
            "  - ONLY compute if no existing data is found AND the user wants new data.\n"
            "  - If no structure found at all, generate one.\n"
            "• 'Generate/Create/Make X' → generate_alloy_supercell IMMEDIATELY.\n"
            "• 'Search/Find/List X' → search_database ONLY.\n"
            "• User confirms 'yes/ok/proceed' → generate_alloy_supercell NOW.\n\n"
            "VIEWING EXISTING DATA:\n"
            "• 'Show/display/view structure/RDF/image of X' → search_database(structure_ref='X') to resolve, then generate_report.\n"
            "• 'Structure 111' or 'structure ID 111' → search_database(structure_ref='111') for direct lookup.\n"
            "  If user wants visualizations, call generate_report(structure_ref=UUID).\n"
            "• NEVER call generate_alloy_supercell to 'show' an existing structure.\n\n"
            "COMPOSITION INPUT:\n"
            "ALWAYS use composition_string parameter. Examples:\n"
            "• 'Cu-Ag structures?' → search_database(composition_string='Cu-Ag')\n"
            "• 'fcc Cu50Ag50' → generate_alloy_supercell(structure='fcc', composition_string='Cu50Ag50')\n"
            "Supported: 'Cu-Zr', 'Ag75Cu25', 'Ag3Cu1' (ratio). Order matters: Ni75Ag25 ≠ Ag75Ni25.\n\n"
            "CALCULATOR FILTERING:\n"
            "When the user mentions a calculator ('using MACE', 'with NequIP', 'orb calculator'), "
            "pass calculator_name to search_database. Shorthands: 'mace', 'orb', 'nequip', or full name.\n"
            "Example: 'elastic properties of CoCrFeNi with MACE' → "
            "search_database(composition_string='CoCrFeNi', calculator_name='mace')\n\n"
            "STABILITY ASSESSMENT:\n"
            "• Structural: If structural_match_percent < 90%, WARN about instability/phase transformation.\n"
            "• Elastic: If born_criterion_satisfied=False, WARN about mechanical instability.\n\n"
            "CACHED DATA PRESENTATION:\n"
            "After search_database returns results:\n"
            "1. If has_elastic_properties=True or has_qha_data=True, present those property values.\n"
            "2. For ANY existing structure, use generate_report(structure_ref=UUID) to show images, RDF, and charts.\n"
            "3. Ask if the user wants a detailed report with visualizations.\n"
            "If yes, call generate_report with the structure UUID.\n\n"
            "CACHED DATA RULE:\n"
            "After calling search_database, if results show existing properties, present them.\n"
            "Do NOT call calculate_elastic_properties or compute_anharmonic_properties on data that already exists.\n"
            "Cached data ALWAYS wins.\n\n"
            "CALCULATOR COMPARISON:\n"
            "• Regenerate (new SQS): 'Generate with NequIP' → change setting, then generate\n"
            "• Benchmark (same atoms): 'Compare MACE vs ORB on THIS structure' → recompute_structure tool\n"
            "If unclear, ASK which approach user wants.\n\n"
            "RULES:\n"
            "• Execute tools sequentially (one at a time)\n"
            "• Use default parameters unless troubleshooting or user specifies otherwise\n"
            "• Ask for missing essential information\n"
            "• CRITICAL: When presenting results or answering questions (NOT when calling tools), "
            "ALWAYS end your response with a follow-up question (?). Examples: "
            "'Would you like to know more?' or 'Shall I run another analysis?' "
            "This is required for the system to work.\n\n"
            "Goal: Make simulation results accessible and actionable for materials researchers."
        ),
        "assistant": (
            "You are a helpful AI assistant. "
            "Answer questions clearly and concisely. "
            "If you don't know something, say so."
        ),
    }

    @staticmethod
    def create_scientist(
        model_client: Any,
        tools: List[Callable],
        name: str = "Scientist",
        system_message: Optional[str] = None,
        model_client_stream: bool = True,
        reflect_on_tool_use: bool = True,
        temperature: float = 0.0,
    ) -> BaseAgent:
        """
        Create a Scientist agent.

        Args:
            model_client: LLM client (e.g., OpenAI client)
            tools: List of tool functions available to the agent
            name: Agent name
            system_message: Custom system message (uses default if None)
            model_client_stream: Enable streaming responses
            reflect_on_tool_use: Enable reflection after tool use
            temperature: Sampling temperature

        Returns:
            Configured ScientistAgent instance

        Examples:
            >>> from autogen_ext.models import OpenAIChatCompletionClient
            >>> client = OpenAIChatCompletionClient(...)  # doctest: +SKIP
            >>> agent = AgentFactory.create_scientist(  # doctest: +SKIP
            ...     model_client=client,
            ...     tools=[generate_alloy_supercell]
            ... )
        """
        config = AgentConfig(
            name=name,
            system_message=system_message or AgentFactory.DEFAULT_MESSAGES["scientist"],
            tools=tools,
            model_client=model_client,
            model_client_stream=model_client_stream,
            reflect_on_tool_use=reflect_on_tool_use,
            temperature=temperature,
        )
        return ScientistAgent(config)

    @staticmethod
    def create_custom(
        name: str,
        system_message: str,
        model_client: Any,
        tools: Optional[List[Callable]] = None,
        model_client_stream: bool = True,
        reflect_on_tool_use: bool = True,
        temperature: float = 0.0,
    ) -> BaseAgent:
        """
        Create a custom agent with specific configuration.

        Args:
            name: Agent name
            system_message: System prompt defining agent behavior
            model_client: LLM client
            tools: Optional list of tool functions
            model_client_stream: Enable streaming
            reflect_on_tool_use: Enable reflection
            temperature: Sampling temperature

        Returns:
            Configured BaseAgent instance

        Examples:
            >>> agent = AgentFactory.create_custom(  # doctest: +SKIP
            ...     name="CustomAgent",
            ...     system_message="You are a helpful assistant",
            ...     model_client=client
            ... )
        """
        from .scientist import ScientistAgent  # Use ScientistAgent as default implementation

        config = AgentConfig(
            name=name,
            system_message=system_message,
            tools=tools or [],
            model_client=model_client,
            model_client_stream=model_client_stream,
            reflect_on_tool_use=reflect_on_tool_use,
            temperature=temperature,
        )
        return ScientistAgent(config)

    @staticmethod
    def build_dynamic_system_message(base_message: str, session_context: str) -> str:
        """
        Prepend session context block to base system message.

        This enables the agent to be aware of current session parameters
        and any changes since the last calculation.

        Args:
            base_message: The base system message for the agent
            session_context: Context block generated by SessionState.generate_context_block()

        Returns:
            Combined system message with context prepended, or original message
            if session_context is empty

        Examples:
            >>> context = "## SESSION CONTEXT\\n- Calculator: orb-v3"
            >>> base = "You are an expert..."
            >>> result = AgentFactory.build_dynamic_system_message(base, context)
            >>> "SESSION CONTEXT" in result
            True
        """
        if not session_context or not session_context.strip():
            return base_message
        return f"{session_context}\n\n---\n\n{base_message}"

    @staticmethod
    def get_default_scientist_message() -> str:
        """
        Get the default scientist system message.

        Returns:
            The default system message for scientist agents

        Examples:
            >>> msg = AgentFactory.get_default_scientist_message()
            >>> "materials science" in msg.lower()
            True
        """
        return AgentFactory.DEFAULT_MESSAGES["scientist"]

    @staticmethod
    def update_agent_config(
        agent: BaseAgent,
        system_message: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Update an existing agent's configuration.

        Args:
            agent: Agent to update
            system_message: New system message (optional)
            tools: New tools list (optional)
            temperature: New temperature (optional)

        Examples:
            >>> AgentFactory.update_agent_config(  # doctest: +SKIP
            ...     agent,
            ...     system_message="New instructions",
            ...     tools=[new_tool]
            ... )
        """
        if system_message is not None:
            agent.update_system_message(system_message)
        if tools is not None:
            agent.update_tools(tools)
        if temperature is not None:
            agent.config.temperature = temperature
            # Force recreation if agent already exists
            if agent._agent_instance is not None:
                agent._agent_instance = agent.create_agent()
