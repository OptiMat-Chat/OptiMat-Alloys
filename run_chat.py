# Standard library imports
from typing import List, cast, Annotated, Dict, Literal, Any, Optional
import asyncio
import threading
import logging
import os
import yaml
import json
from pathlib import Path

#######################
# LOGGING CONFIGURATION
#######################

# Configure logging verbosity for AutoGen and other libraries
# IMPORTANT: Must be configured BEFORE importing AutoGen/Chainlit to take effect
# By default, suppress verbose DEBUG/INFO output from AutoGen's internal loggers
# Set LOG_LEVEL=DEBUG in .env to enable verbose logging for debugging
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()

# Set up Python's root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress verbose AutoGen loggers by name (before importing)
# These loggers output message publishing, serialization, and agent response details
logging.getLogger("autogen_agentchat").setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))
logging.getLogger("autogen_agentchat.events").setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))
logging.getLogger("autogen_core").setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))
logging.getLogger("autogen").setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

# Suppress other noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Third-party imports (after logging configuration)
import chainlit as cl
from chainlit.input_widget import Select,Slider,Switch,Tags,TextInput

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient

####################
# GLOBAL VARIABLES #
####################

# All structures stored in centralized database (structures/database.json)
# Individual structure files in structures/{id}/ directories

####################
# HELPER FUNCTIONS #
####################

from ase import Atoms
from ase.calculators.calculator import Calculator

# Import core modules (refactored in Phase 1, 2 & 3)
from src.core.calculators import load_calculator, CalculatorManager
from src.core.structure_builder import (
    AlloyBuilder,
    estimate_alloy_lattice_constant,
    lattice_constant_from_atomic_volume,
    compute_replication_factors
)
from src.core.optimization import StructureOptimizer, relax_atoms
from src.core.sqs import SQSGenerator
from src.core.analysis import structural_analysis, compute_density, compute_coordination_rdf
from src.core.elasticity import compute_elastic_stiffness_tensor, compute_elastic_moduli
from src.core.reference_data import (
    initial_cell,
    extract_lattice_constant_a,
    precompute_and_save,
    load_reference_energies,
    ReferenceMode
)
from src.core.formation_energy import formation_energy_per_atom
from src.core.cancellation import ComputationCancelledException

# Import visualization modules (refactored in Phase 2)
from src.visualization.ovito_renderer import render_structure, render_atoms, render_trajectory
from src.visualization.plotly_charts import plot_structural_analysis, plot_coordination_rdf, plot_stiffness_tensor_heatmap
from src.visualization.database_charts import (
    create_growth_curve,
    create_element_usage_chart,
    create_calculator_distribution,
    create_structure_type_distribution,
    create_composition_complexity_distribution,
    create_supercell_size_distribution,
    create_property_availability_heatmap
)

# Import storage modules (refactored in Phase 3 & 4)
from src.storage.database import StructureDatabase, create_structure_database
from src.storage.cache import ReferenceDataCache, get_reference_cache

# Import agent modules (refactored in Phase 3)
from src.agents.factory import AgentFactory

# Import tool modules (refactored in Phase 7)
from src.tools import (
    generate_alloy_supercell,
    search_database,
    generate_report,  # Combined visual report + PDF/data export
    calculate_elastic_properties,
    visualize_database_statistics,
    compute_anharmonic_properties,
    recompute_structure,
)
from src.tools.database_statistics import visualize_database_statistics_internal

# Import utility modules (API key management, environment configuration, session state)
from src.utils.env_manager import update_env_variable, can_write_env_file
# API validation removed (OpenAI provider not used in Docker build)
from src.utils.session_state import SessionState

# Import Ollama utilities (for local model support)
from src.utils.ollama_manager import (
    is_ollama_installed,
    ensure_ollama_running,
    get_available_ollama_models,
    get_recommended_models,
    is_model_installed,
    pull_model,
    pull_model_with_progress,
    get_model_vram_gb,
)

# Import Ollama configuration (for preferred model constant)
from src.agents.local_models.ollama_config import PREFERRED_OLLAMA_MODEL

# Import unified model factory (supports OpenRouter and Ollama)
from src.agents.model_factory import (
    create_unified_model_client,
    get_unified_model_info,
    get_all_available_models,
    parse_model_selection,
    has_openrouter_key,
    is_free_model,
)

# Phase 1, 2 & 3 Refactoring - Functions moved to modules:
# Phase 1:
#   - load_calculator → src/core/calculators.py
#   - AlloyBuilder, estimate_alloy_lattice_constant → src/core/structure_builder.py
#   - relax_atoms → src/core/optimization.py
#   - SQSGenerator → src/core/sqs.py
# Phase 2:
#   - structural_analysis, compute_density → src/core/analysis.py
#   - render_structure, render_atoms, render_trajectory → src/visualization/ovito_renderer.py
#   - plot_structural_analysis, plot_coordination_rdf → src/visualization/plotly_charts.py
# Phase 3:
#   - initial_cell, extract_lattice_constant_a, precompute_and_save, load_reference_energies → src/core/reference_data.py
#   - formation_energy_per_atom → src/core/formation_energy.py
#   - AgentFactory, ScientistAgent → src/agents/
#   - GlobalStructureDatabase → src/storage/global_database.py


###############
# AGENT TOOLS #
###############

# Tool: generate_alloy_supercell moved to src/tools/alloy_generation.py
# Tool: search_database moved to src/tools/database_search.py

# Tool: generate_report moved to src/tools/generate_report.py (combined visual + export)
# Tool: calculate_elastic_properties moved to src/tools/elastic_properties.py
# Tool: visualize_database_statistics moved to src/tools/database_statistics.py
################
# END OF TOOLS #
################

logger = logging.getLogger(__name__)


@cl.on_app_startup
async def setup_files_directory():
    """Ensure .files/ exists — Chainlit's shutdown handler deletes it."""
    from pathlib import Path
    from chainlit.config import FILES_DIRECTORY
    Path(FILES_DIRECTORY).mkdir(parents=True, exist_ok=True)


@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    import datetime
    session_id = cl.user_session.get("id")  # type: ignore
    print(f"[{datetime.datetime.now()}] SESSION STARTED - Session ID: {session_id}")

    #############################
    # API KEY MANAGEMENT
    #############################
    # Check for existing Ollama API key (needed for cloud models — the default)
    existing_ollama_key = os.getenv("OLLAMA_API_KEY", "").strip()

    if not existing_ollama_key or existing_ollama_key.startswith("your-"):
        # Prompt user for Ollama API key (required for cloud model — the default)
        res = await cl.AskUserMessage(
            content=(
                "🔑 **Ollama API Key Required**\n\n"
                "The default model (`gpt-oss:120b-cloud`) runs via Ollama Cloud and requires an API key.\n\n"
                "**How to get your API key:**\n"
                "1. Sign up at https://ollama.com/\n"
                "2. Go to https://ollama.com/settings/keys\n"
                "3. Click **\"Keys\"** in the left panel, then click **\"Add API Key\"**\n"
                "4. Copy the key and paste it below\n\n"
                "**Note:** After entering your key, you'll also need to authorize this device (one-time step — just click \"Connect\" in your browser)."
            ),
            timeout=300
        ).send()

        if not res or not res.get('output'):
            await cl.Message(
                content="❌ **Ollama API key is required for cloud models.**\n\n"
                        "Please refresh the page and enter your API key, or select an OpenRouter model in Settings."
            ).send()
            return

        ollama_key = res['output'].strip()
        os.environ["OLLAMA_API_KEY"] = ollama_key

        # Try to save to .env file
        can_write, write_msg = can_write_env_file()
        if can_write:
            success, update_msg = update_env_variable("OLLAMA_API_KEY", ollama_key)
            if success:
                await cl.Message(content="✅ Ollama API key saved for future sessions.").send()
            else:
                await cl.Message(content="✅ Ollama API key set for this session.").send()
        else:
            await cl.Message(content="✅ Ollama API key set for this session.").send()
    else:
        print(f"[{datetime.datetime.now()}] Using existing OLLAMA_API_KEY from environment")

    # Get database path
    db = create_structure_database()
    db_path = db.get_database_path()

    # Build unified model dropdown at startup
    # This combines: 1) OpenRouter free models, 2) Ollama models
    available_models = await get_all_available_models()
    model_values = [m["id"] for m in available_models]

    # First available model is the default
    initial_model_index = 0

    # Build tooltip based on available providers
    tooltip_parts = ["OpenRouter: Free cloud models"]
    tooltip_parts.append("Ollama: Local & cloud models")
    model_tooltip = " | ".join(tooltip_parts)

    # Send settings with unified model dropdown
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="AI Model",
                values=model_values,
                initial_index=initial_model_index,
                tooltip=model_tooltip
            ),
            Select(
                id="Calculator",
                label="Force Field Calculator",
                values=[
                    # ORB (Orbital Materials) - default
                    "orb-v3-conservative-inf-omat",
                    "orb-v3-direct-20-omat",
                    # NequIP Foundation Models (run in separate optimat-nequip environment)
                    "nequip-oam-l",      # F1=0.893, trained on OMat24+MPtrj+sAlex
                    "nequip-oam-xl",     # F1=0.906, highest accuracy
                    "nequip-mp-l",       # F1=0.761, trained on MPtrj only
                    # MACE-MPA-0 (Matbench SOTA)
                    "mace-mpa-0-medium",
                    # MACE-OMAT-0 (best for phonons)
                    "mace-omat-0-medium",
                    "mace-omat-0-small",
                ],
                initial_index=0,  # ORB conservative: most accurate (default)
                tooltip="ORB: Orbital Materials | NequIP: Foundation Models | MACE-MPA/OMAT: Matbench SOTA/Phonons"
            ),
            Select(
                id="SupercellSize",
                label="Default Supercell Size",
                values=["Small (48 atoms)", "Medium (512 atoms)", "Large (2048 atoms)"],
                initial_index=1,  # Medium (512 atoms): balanced (default)
                tooltip="Small (48): Fastest, initial exploration | Medium (512): Balanced | Large (2048): Best statistics, slower"
            ),
        ]
    ).send()

    # Get calculator from settings
    calculator = settings["Calculator"]
    cl.user_session.set("default_calculator", calculator)

    # Get supercell size from settings and convert to atom count
    supercell_size_label = settings["SupercellSize"]
    supercell_size_map = {
        "Small (48 atoms)": 48,
        "Medium (512 atoms)": 512,
        "Large (2048 atoms)": 2048
    }
    supercell_size = supercell_size_map.get(supercell_size_label, 48)  # Default to 48 if not found
    cl.user_session.set("default_supercell_size", supercell_size)

    # Initialize session state for memory layer (parameter change detection)
    session_state = SessionState()
    session_state.update_current_params(calculator=calculator, supercell_size=supercell_size)
    cl.user_session.set("session_state", session_state)

    # Use cache to check if reference data is available for selected calculator
    cache = get_reference_cache(calculator=calculator)
    if not cache.is_available():
        await cl.Message(
        content=(
            f"**Precomputing lattice constants and energies of elements with {calculator}. This is the one-time setup, please wait and do not disrupt the calculations!**\n\n"
        )
        ).send()
        precompute_and_save(
            hydrostatic_cell_relaxation=True,
            optimizer="FIRE",
            fmax=0.005,
            calculator=calculator,
            cache=cache,
        )
    else:
        print(f"Reference data cache available for {calculator} — skipping precompute_and_save.")

    # IF THIS MESSAGE IS SENT, THE STARTERS ARE NOT WORKING ANYMORE, HAS TO SEE WHAT IS MORE PREFERABLE
    # MAYBE MOVE THIS CHUNK OF CODE TO THE MESSAGES AND LIMIT IT TO THE FISRT USER MESSAGE
    # MAYBE EVEN IMPROVE THE CODE ABOVE TO LET LLM GENERATE MEANINGFUL FOLDER NAME FROM THEIR FIRST REQUEST
    import random

    # Separate phrases into statements and questions
    encouraging_statements = [
        "Let's discover something amazing today! 🚀",
        "Ready to push the boundaries of materials science! 🔬",
        "Your next big breakthrough starts here. ✨",
        "Time to turn atoms into answers! 🧪",
        "Let's explore the unknown, one simulation at a time. 🌌",
        "Curiosity is the best catalyst — let's get started! ⚡",
        "Science waits for no one — let's dive in! 🌊",
        "From quanta to quality — let's make it happen! 📈",
        "Every great material starts with a great idea. 💡"
    ]

    encouraging_questions = [
        "Shall we create the next wonder material? 🏆",
        "What groundbreaking material will we uncover today? 🧩",
        "Are you ready to make a discovery? 🔍"
    ]

    # Choose whether to pull from statements or questions
    if random.random() < 0.5:  # 50% chance to use a statement
        phrase = random.choice(encouraging_statements)
        followup = "\n\n✨ **Which material do we start with?**"
    else:
        phrase = random.choice(encouraging_questions)
        followup = ""  # No second question

    # Parse unified model selection
    model_selection = settings["Model"]
    provider, model_name = parse_model_selection(model_selection)

    # Default fallback model (Ollama cloud — most reliable, no provider rate limits)
    FALLBACK_PROVIDER = "ollama"
    FALLBACK_MODEL = "gpt-oss:120b-cloud"

    # Handle Ollama model selection: ensure server running, auto-pull if needed
    if provider == "ollama":
        from src.agents.local_models.ollama_config import OLLAMA_MODELS
        is_cloud_model = OLLAMA_MODELS.get(model_name, {}).get("cloud", False)

        # Ensure Ollama server is running
        success, message = await ensure_ollama_running(timeout=30)

        if not success:
            await cl.Message(
                content=f"**Ollama Error**: {message}\n\nFalling back to OpenRouter free model..."
            ).send()
            provider = FALLBACK_PROVIDER
            model_name = FALLBACK_MODEL
        elif is_cloud_model:
            # Cloud models need either a saved device key (~/.ollama/id_ed25519,
            # written by `ollama login`) or an OLLAMA_API_KEY. If neither is
            # present, prompt the user to authorize. If a real 401 occurs later,
            # the runtime error handler shows the same guidance.
            device_key = Path.home() / ".ollama" / "id_ed25519"
            if not (device_key.exists() or os.getenv("OLLAMA_API_KEY")):
                import subprocess
                login_result = subprocess.run(
                    ["ollama", "login"], capture_output=True, text=True, timeout=10
                )
                login_url = ""
                for line in (login_result.stdout + login_result.stderr).split("\n"):
                    if "https://ollama.com/connect" in line:
                        login_url = line.strip()
                        break

                await cl.Message(
                    content=f"🔑 **Ollama Cloud — Device Authorization Required**\n\n"
                            f"The cloud model `{model_name}` requires a one-time device authorization.\n\n"
                            f"**Steps:**\n"
                            f"1. [Click here to authorize this device]({login_url})\n"
                            f"2. Log in to your Ollama account (or sign up at https://ollama.com/)\n"
                            f"3. Click **\"Connect\"** to authorize this device\n\n"
                            f"**If you don't have an API key yet:**\n"
                            f"1. Go to https://ollama.com/settings/keys\n"
                            f"2. Click **\"Add API Key\"**\n"
                            f"3. Copy the key and enter it when prompted\n\n"
                            f"After authorization, refresh the page. This only needs to be done once.\n\n"
                            f"Falling back to alternative model for now..."
                ).send()
                provider = FALLBACK_PROVIDER
                model_name = FALLBACK_MODEL
        else:
            # Local models — check GPU capability (warning only, not blocking)
            # Ollama can offload model layers to system RAM, so it may work even
            # with insufficient VRAM, just slower.
            from src.utils.platform import check_cuda_availability, get_cuda_memory_info
            required_vram = OLLAMA_MODELS.get(model_name, {}).get("vram_gb", 0)

            if required_vram > 0:
                cuda_available, gpu_name = check_cuda_availability()
                if cuda_available:
                    total_vram_mb, _ = get_cuda_memory_info()
                    total_vram_gb = total_vram_mb / 1024

                    if total_vram_gb < required_vram:
                        await cl.Message(
                            content=f"⚠️ **Limited GPU Memory**\n\n"
                                    f"The model `{model_name}` recommends **{required_vram}GB VRAM**, "
                                    f"but your GPU (`{gpu_name}`) has **{total_vram_gb:.1f}GB**.\n\n"
                                    f"The model may still work (Ollama offloads to system RAM), but performance will be slower.\n\n"
                                    f"For best performance, consider:\n"
                                    f"1. `gpt-oss:120b-cloud` (cloud, no GPU needed)\n"
                                    f"2. OpenRouter free models (cloud, no GPU needed)"
                        ).send()
                    else:
                        await cl.Message(
                            content=f"✓ GPU check passed: `{gpu_name}` ({total_vram_gb:.1f}GB VRAM) — "
                                    f"sufficient for `{model_name}` ({required_vram}GB recommended)"
                        ).send()
                else:
                    await cl.Message(
                        content=f"⚠️ **No GPU detected**\n\n"
                                f"The local model `{model_name}` requires a GPU with **{required_vram}GB VRAM** "
                                f"but no GPU was found on this device.\n\n"
                                f"Switching to cloud model `gpt-oss:120b-cloud` (no GPU needed)."
                    ).send()
                    provider = FALLBACK_PROVIDER
                    model_name = FALLBACK_MODEL

            # Pull model if not installed (only if we haven't fallen back to cloud)
            if provider == "ollama" and not is_model_installed(model_name):
                if model_name == PREFERRED_OLLAMA_MODEL:
                    progress_msg = cl.Message(
                        content=f"⏳ **Installing `{model_name}`** — this is a one-time download..."
                    )
                    await progress_msg.send()

                    last_reported = -1
                    pull_success = False
                    async for status, percent in pull_model_with_progress(model_name):
                        if "error" in status.lower():
                            break
                        if percent >= 0 and percent // 10 > last_reported:
                            last_reported = percent // 10
                            progress_msg.content = (
                                f"⏳ **Installing `{model_name}`** — downloading: **{percent}%**"
                            )
                            await progress_msg.update()
                        if "success" in status.lower():
                            pull_success = True

                    if not pull_success:
                        await cl.Message(
                            content=f"❌ **Failed to install `{model_name}`**.\n\n"
                                    "Falling back to cloud model..."
                        ).send()
                        provider = FALLBACK_PROVIDER
                        model_name = FALLBACK_MODEL
                    else:
                        progress_msg.content = f"✅ **`{model_name}` is ready!** The model is installed and won't need downloading again."
                        await progress_msg.update()
                else:
                    # Non-preferred model not installed
                    await cl.Message(
                        content=f"**Model not installed**: `{model_name}`\n\n"
                                f"Please run `ollama pull {model_name}` and try again.\n\n"
                                "Falling back to cloud model..."
                    ).send()
                    provider = FALLBACK_PROVIDER
                    model_name = FALLBACK_MODEL

    # Store provider and model in session
    cl.user_session.set("current_provider", provider)
    cl.user_session.set("current_model", model_name)

    # Create model client using unified factory
    model_client = create_unified_model_client(provider, model_name, temperature=0.0)

    # Now show welcome messages with correct provider/model info
    # Create welcome image element
    title_image = cl.Image(path="public/title_image.png", name="OptiMat Alloys Title", display="inline")

    # Get database count and statistics for Message 2
    db_count = db.count()
    db_stats = await visualize_database_statistics_internal()

    # Message 1: Welcome header with title image
    try:
        await cl.Message(
            content="**Welcome, Explorer! Your AI research partner is online.**",
            elements=[title_image]
        ).send()
    except Exception:
        logger.debug("Title image send failed, sending welcome without it")
        await cl.Message(
            content="**Welcome, Explorer! Your AI research partner is online.**",
        ).send()

    # Format provider/model display
    if provider == "ollama":
        provider_display = "☁️ Ollama (Cloud)" if "cloud" in model_name else "🖥️ Ollama (Local)"
        model_display = f"`{model_name}`"
    else:
        provider_display = "🆓 OpenRouter (Free)"
        model_display = f"`{model_name}`"

    # Message 2: Settings and database statistics (auto-displayed)
    chart_names = " ".join(db_stats.get('chart_types', []))
    await cl.Message(
        content=(
            f"**⚙️ Current Settings**\n\n"
            f"🤖 **LLM Provider:** {provider_display}\n"
            f"📝 **Model:** {model_display}\n"
            f"📂 **Database:** `{db_path}`\n"
            f"🔧 **ORB Calculator:** `{calculator}`\n"
            f"🔢 **Default Supercell Size:** {supercell_size_label}\n\n"
            f"---\n\n"
            f"**Database Statistics**\n\n"
            f"Your database contains **{db_count} structures**.\n\n"
            f"{chart_names}"
        ),
        elements=db_stats.get('elements', [])
    ).send()

    # Show data storage notice for first-time users (empty database)
    if db_count == 0:
        await cl.Message(
            content=(
                "💾 **Your data is saved automatically.**\n\n"
                "All structures and calculations are stored inside this app.\n"
                "To keep your data between sessions:\n"
                "- ✅ Use the **Stop** button in Docker Desktop to pause the app\n"
                "- ✅ Use the **Start** button to resume — your data will still be there\n"
                "- ❌ Do NOT click **Delete** — this will remove the app and all your data\n\n"
                "You can export your database at any time using the button below."
            ),
            actions=[cl.Action(name="download_database", label="📦 Download Database", payload={})]
        ).send()
    else:
        # For returning users with data, show download button in a compact message
        await cl.Message(
            content=f"💾 **{db_count} structures** in your database.",
            actions=[cl.Action(name="download_database", label="📦 Download Database", payload={})]
        ).send()

    # Message 3: Rotating encouragement/questions
    await cl.Message(
        content=(
            f"*{phrase}*"
            f"{followup}"
        )
    ).send()

    # Show Ollama hardware note if using a local Ollama model (skip for cloud-proxied models)
    if provider == "ollama" and "cloud" not in model_name:
        vram_required = get_model_vram_gb(model_name)
        if vram_required:
            vram_recommended = int(vram_required) + 4
            hw_note = (
                f"This model needs **~{vram_required} GB VRAM** for full GPU acceleration. "
                f"**{vram_recommended} GB or more is recommended** for good performance. "
                f"Less powerful devices will still work, but responses will be slower as "
                f"Ollama offloads layers to system RAM."
            )
        else:
            hw_note = (
                "VRAM requirement depends on the model size. Ollama will offload layers "
                "to system RAM if your GPU runs short, but responses will be slower."
            )
        await cl.Message(
            content=(
                f"🖥️ **Using Ollama Local Model: `{model_name}`**\n\n"
                f"{hw_note}\n\n"
                f"💡 Prefer no-VRAM cloud inference? Switch to `gpt-oss:120b-cloud` in ⚙️ Settings."
            )
        ).send()

    # Create the Scientist agent using AgentFactory
    # Note: reflect_on_tool_use=False to avoid AutoGen bug (issue #6328)
    # Tool result interpretation is achieved through enhanced system message

    # Tools available to the agent
    tools_list = [
        generate_alloy_supercell,
        search_database,
        generate_report,  # Combined visual report + PDF/data export
        calculate_elastic_properties,
        visualize_database_statistics,
        compute_anharmonic_properties,  # QHA: temperature-dependent properties
        recompute_structure,  # Recompute existing structure with different calculator
    ]

    # Store the base system message for dynamic context injection
    base_system_message = AgentFactory.get_default_scientist_message()
    cl.user_session.set("base_system_message", base_system_message)

    scientist_agent = AgentFactory.create_scientist(
        model_client=model_client,
        tools=tools_list,
        name="Scientist",
        model_client_stream=True,
        reflect_on_tool_use=False,  # Use prompt engineering for reflection
    )
    scientist = scientist_agent.get_agent()

    # Store the scientist agent wrapper for dynamic system message updates
    cl.user_session.set("scientist_agent_wrapper", scientist_agent)

    # Define a custom termination condition:
    # Terminate the group chat if any message from Scientist includes a "?".
    termination = TextMentionTermination("?", sources=["Scientist"])

    # Chain the Scientist using RoundRobinGroupChat.
    group_chat = RoundRobinGroupChat(
        [scientist],
        max_turns=10,  # Prevent infinite reasoning loops
        termination_condition=termination
    )

    # Set up the user session with the group chat.
    cl.user_session.set("prompt_history", "")  # type: ignore
    cl.user_session.set("team", group_chat)      # type: ignore

    # Create cancellation token for stop button (AutoGen agent-level cancellation)
    cancellation_token = CancellationToken()
    cl.user_session.set("cancellation_token", cancellation_token)  # type: ignore

    # Create computation cancellation event for stop button (computation-level cancellation)
    # This event is used for cooperative cancellation of long-running computations
    # like elastic tensor calculations. Tools check this event periodically and stop
    # gracefully when it's set.
    computation_cancellation_event = threading.Event()
    cl.user_session.set("computation_cancellation_event", computation_cancellation_event)  # type: ignore


@cl.action_callback("download_database")  # type: ignore
async def on_download_database(action: cl.Action):
    """Handle 'Download Database' button click — send database.db as downloadable file."""
    db = create_structure_database()
    db_path = db.get_database_path()

    if not os.path.exists(db_path):
        await cl.Message(content="❌ No database file found.").send()
        return

    db_count = db.count()
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    await cl.Message(
        content=f"📦 **Database Export** — {db_count} structures ({file_size_mb:.1f} MB)",
        elements=[cl.File(name="database.db", path=db_path, display="inline")]
    ).send()


@cl.on_settings_update  # type: ignore
async def on_settings_update(settings: Dict[str, Any]) -> None:
    """
    Handle settings updates during an active session.

    Updates provider, calculator, and model selection dynamically. Calculator changes
    affect future structures only (immutable structures approach). Provider and model
    changes take effect immediately for all subsequent agent interactions.
    """
    import datetime
    try:
        print(f"[{datetime.datetime.now()}] SETTINGS UPDATE - New settings: {settings}")

        # Handle calculator changes
        new_calculator = settings.get("Calculator")
        old_calculator = cl.user_session.get("default_calculator")  # type: ignore

        if new_calculator and new_calculator != old_calculator:
            # Update session calculator
            cl.user_session.set("default_calculator", new_calculator)  # type: ignore

            # Update session state for memory layer
            session_state = cl.user_session.get("session_state")  # type: ignore
            if session_state:
                current_supercell = cl.user_session.get("default_supercell_size", 512)  # type: ignore
                session_state.update_current_params(
                    calculator=new_calculator,
                    supercell_size=current_supercell
                )

            # Notify user about the change
            await cl.Message(
                content=(
                    f"**Calculator changed**: `{old_calculator}` → `{new_calculator}`\n\n"
                    f"**Note**: This change will affect future structure calculations only. "
                    f"Existing structures remain unchanged (immutable structures approach)."
                )
            ).send()

            # Check if reference data is available for new calculator
            cache = get_reference_cache(calculator=new_calculator)
            if not cache.is_available():
                await cl.Message(
                    content=(
                        f"**Reference data not available for {new_calculator}.**\n\n"
                        f"Reference data will be precomputed automatically when you create your first structure with this calculator."
                    )
                ).send()

        # Handle model changes (unified approach for OpenAI, OpenRouter, and Ollama)
        new_model_selection = settings.get("Model")
        old_provider = cl.user_session.get("current_provider")  # type: ignore
        old_model = cl.user_session.get("current_model")  # type: ignore

        if new_model_selection:
            new_provider, new_model = parse_model_selection(new_model_selection)

            # Check if model or provider actually changed
            if new_provider != old_provider or new_model != old_model:
                # Handle OpenRouter model: check for API key
                if new_provider == "openrouter":
                    existing_or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
                    if not existing_or_key or existing_or_key.startswith("your-"):
                        res = await cl.AskUserMessage(
                            content=(
                                "🔑 **OpenRouter API Key Required**\n\n"
                                "To use OpenRouter free models, you need an API key.\n\n"
                                "**How to get your key:**\n"
                                "1. Go to https://openrouter.ai/ and click **\"Get API Key\"**\n"
                                "2. Sign in or create an account\n"
                                "3. Click **\"API Keys\"** in the left sidebar\n"
                                "4. Click **\"Create\"** to generate a new key\n"
                                "5. Copy the key and paste it below\n\n"
                                "**Note:** Free models are available without adding credits.\n\n"
                                "**Tip:** We recommend depositing $10+ on your OpenRouter account to unlock "
                                "higher rate limits (1,000 req/day instead of 50). This will **not** be spent "
                                "on free models — it just removes the rate limit cap."
                            ),
                            timeout=300
                        ).send()

                        if not res or not res.get('output'):
                            await cl.Message(
                                content="❌ OpenRouter API key is required. Keeping current settings."
                            ).send()
                            return

                        or_key = res['output'].strip()
                        os.environ["OPENROUTER_API_KEY"] = or_key

                        can_write, _ = can_write_env_file()
                        if can_write:
                            success, _ = update_env_variable("OPENROUTER_API_KEY", or_key)
                            if success:
                                await cl.Message(content="✅ OpenRouter API key saved.").send()
                            else:
                                await cl.Message(content="✅ OpenRouter API key set for this session.").send()
                        else:
                            await cl.Message(content="✅ OpenRouter API key set for this session.").send()

                # Handle Ollama model: ensure server running, auto-pull if needed
                if new_provider == "ollama":
                    from src.agents.local_models.ollama_config import OLLAMA_MODELS
                    is_cloud_model = OLLAMA_MODELS.get(new_model, {}).get("cloud", False)

                    success, message = await ensure_ollama_running(timeout=30)
                    if not success:
                        await cl.Message(
                            content=f"**Ollama Error**: {message}\n\nKeeping current settings."
                        ).send()
                        return

                    # Cloud models need device authorization check
                    if is_cloud_model:
                        device_key = Path.home() / ".ollama" / "id_ed25519"
                        if not (device_key.exists() or os.getenv("OLLAMA_API_KEY")):
                            import subprocess
                            login_result = subprocess.run(
                                ["ollama", "login"], capture_output=True, text=True, timeout=10
                            )
                            login_url = ""
                            for line in (login_result.stdout + login_result.stderr).split("\n"):
                                if "https://ollama.com/connect" in line:
                                    login_url = line.strip()
                                    break
                            await cl.Message(
                                content=f"🔑 **Ollama Cloud — Device Authorization Required**\n\n"
                                        f"1. Open this URL in your browser:\n"
                                        f"   {login_url}\n"
                                        f"2. Log in to your Ollama account\n"
                                        f"3. Click **\"Connect\"** to authorize this device\n\n"
                                        f"**Need an API key?** Go to https://ollama.com/settings/keys → click **\"Keys\"** → **\"Add API Key\"**\n\n"
                                        f"Then select the model again. Keeping current settings."
                            ).send()
                            return

                    # Local models — check GPU capability before pulling
                    if not is_cloud_model:
                        from src.utils.platform import check_cuda_availability, get_cuda_memory_info
                        required_vram = OLLAMA_MODELS.get(new_model, {}).get("vram_gb", 0)

                        if required_vram > 0:
                            cuda_available, gpu_name = check_cuda_availability()
                            if cuda_available:
                                total_vram_mb, _ = get_cuda_memory_info()
                                total_vram_gb = total_vram_mb / 1024
                                if total_vram_gb < required_vram:
                                    # Warning only — Ollama can offload layers to system RAM
                                    await cl.Message(
                                        content=f"⚠️ **Limited GPU Memory** — `{new_model}` recommends **{required_vram}GB VRAM**, "
                                                f"but `{gpu_name}` has **{total_vram_gb:.1f}GB**.\n\n"
                                                f"The model may still work (Ollama offloads to system RAM), but performance will be slower.\n"
                                                f"For best performance, consider using `gpt-oss:120b-cloud` (no GPU needed) or OpenRouter free models."
                                    ).send()
                            else:
                                await cl.Message(
                                    content=f"⚠️ **No GPU detected** — `{new_model}` requires **{required_vram}GB VRAM** "
                                            f"but no GPU was found.\n\n"
                                            f"Please use `gpt-oss:120b-cloud` (no GPU needed) or OpenRouter free models."
                                ).send()
                                return

                        # Pull model if not installed
                        if not is_model_installed(new_model):
                            if new_model == PREFERRED_OLLAMA_MODEL:
                                progress_msg = cl.Message(
                                    content=f"⏳ **Installing `{new_model}`** — this is a one-time download..."
                                )
                                await progress_msg.send()

                                last_reported = -1
                                pull_success = False
                                async for status, percent in pull_model_with_progress(new_model):
                                    if "error" in status.lower():
                                        break
                                    if percent >= 0 and percent // 10 > last_reported:
                                        last_reported = percent // 10
                                        progress_msg.content = (
                                            f"⏳ **Installing `{new_model}`** — downloading: **{percent}%**"
                                        )
                                        await progress_msg.update()
                                    if "success" in status.lower():
                                        pull_success = True

                                if not pull_success:
                                    await cl.Message(
                                        content=f"❌ **Failed to install `{new_model}`**.\n\n"
                                                f"Please try again or use a different model."
                                    ).send()
                                    return
                                progress_msg.content = f"✅ **`{new_model}` is ready!** The model is installed and won't need downloading again."
                                await progress_msg.update()
                            else:
                                await cl.Message(
                                    content=f"**Model not installed**: `{new_model}`\n\n"
                                            f"Please run `ollama pull {new_model}` and try again, or select a different model."
                                ).send()
                                return

                # Create new model client
                model_client = create_unified_model_client(new_provider, new_model, temperature=0.0)

                # Rebuild tools list (consistent with initialization)
                tools_list = [
                    generate_alloy_supercell,
                    search_database,
                    generate_report,  # Combined visual report + PDF/data export
                    calculate_elastic_properties,
                    visualize_database_statistics,
                    compute_anharmonic_properties,
                    recompute_structure,
                ]

                # Recreate agent with new model client
                scientist_agent = AgentFactory.create_scientist(
                    model_client=model_client,
                    tools=tools_list,
                    name="Scientist",
                    model_client_stream=True,
                    reflect_on_tool_use=False
                )
                scientist = scientist_agent.get_agent()

                # Recreate team with new agent
                termination = TextMentionTermination("?", sources=["Scientist"])
                group_chat = RoundRobinGroupChat(
                    [scientist],
                    max_turns=10,
                    termination_condition=termination
                )

                # Update session (including scientist wrapper for memory layer)
                cl.user_session.set("current_provider", new_provider)  # type: ignore
                cl.user_session.set("current_model", new_model)  # type: ignore
                cl.user_session.set("team", group_chat)  # type: ignore
                cl.user_session.set("scientist_agent_wrapper", scientist_agent)  # type: ignore

                # Build notification message
                is_cloud_ollama = new_provider == "ollama" and "cloud" in new_model
                if new_provider == "ollama":
                    provider_display = "☁️ Ollama (Cloud)" if is_cloud_ollama else "🖥️ Ollama (Local)"
                else:
                    provider_display = "🆓 OpenRouter (Free)"
                model_desc = get_unified_model_info(new_provider, new_model)

                await cl.Message(
                    content=(
                        f"🤖 **Model changed**: `{old_model}` → `{new_model}`\n\n"
                        f"**Provider**: {provider_display}\n"
                        f"**Description**: {model_desc}\n\n"
                        f"ℹ️ Conversation history was cleared for the new model — "
                        f"this prevents tool-call messages from the previous provider "
                        f"from confusing the new one."
                    )
                ).send()

                # Show Ollama hardware note only for local models (skip cloud-proxied)
                if new_provider == "ollama" and not is_cloud_ollama:
                    vram_required = get_model_vram_gb(new_model)
                    if vram_required:
                        vram_recommended = int(vram_required) + 4
                        hw_note = (
                            f"This model needs **~{vram_required} GB VRAM** for full GPU acceleration. "
                            f"**{vram_recommended} GB or more is recommended** for good performance. "
                            f"Less powerful devices will still work, but responses will be slower as "
                            f"Ollama offloads layers to system RAM."
                        )
                    else:
                        hw_note = (
                            "VRAM requirement depends on the model size. Ollama will offload layers "
                            "to system RAM if your GPU runs short, but responses will be slower."
                        )
                    await cl.Message(
                        content=(
                            f"🖥️ **Using Ollama Local Model: `{new_model}`**\n\n"
                            f"{hw_note}\n\n"
                            f"💡 Prefer no-VRAM cloud inference? Switch to `gpt-oss:120b-cloud` in ⚙️ Settings."
                        )
                    ).send()

        # Handle supercell size changes
        new_size_label = settings.get("SupercellSize")
        old_size = cl.user_session.get("default_supercell_size")  # type: ignore

        if new_size_label:
            # Map label to atom count
            supercell_size_map = {
                "Small (48 atoms)": 48,
                "Medium (512 atoms)": 512,
                "Large (2048 atoms)": 2048
            }
            new_size = supercell_size_map.get(new_size_label, 48)

            if new_size != old_size:
                # Update session
                cl.user_session.set("default_supercell_size", new_size)  # type: ignore

                # Update session state for memory layer
                session_state = cl.user_session.get("session_state")  # type: ignore
                if session_state:
                    current_calculator = cl.user_session.get("default_calculator", "orb-v3-direct-20-omat")  # type: ignore
                    session_state.update_current_params(
                        calculator=current_calculator,
                        supercell_size=new_size
                    )

                # Notify user
                await cl.Message(
                    content=(
                        f"🔢 **Default supercell size changed**: {old_size} → {new_size} atoms\n\n"
                        f"📌 **Note**: This change will affect future structure calculations only."
                    )
                ).send()

    except Exception as e:
        import traceback
        print(f"[{datetime.datetime.now()}] ERROR in settings update: {e}")
        traceback.print_exc()
        await cl.Message(content=f"❌ **Error updating settings**: {str(e)}").send()


def _is_provider_error(error_str: str) -> bool:
    """Check if error is a known OpenRouter/cloud provider error."""
    s = error_str.lower()
    return any([
        "429" in s and "provider" in s,
        "rate limit" in s and "provider" in s,
        "403" in s and "run out of credit" in s,
        "402" in s and "insufficient" in s,
        "operation was aborted" in s,
        "internal server error" in s and "responseerror" in s,
    ])


def _get_provider_error_reason(error_str: str) -> str:
    """Extract a user-friendly reason from a provider error."""
    s = error_str.lower()
    if "429" in s or "rate limit" in s:
        return "rate-limited by provider"
    if "403" in s and "run out of credit" in s:
        return "provider capacity exhausted"
    if "402" in s and "insufficient" in s:
        return "provider spend limit reached"
    if "operation was aborted" in s:
        return "provider connection interrupted"
    if "internal server error" in s:
        return "cloud server internal error"
    return "provider error"


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    try:
        # Retrieve the group chat team from the user session.
        team = cast(RoundRobinGroupChat, cl.user_session.get("team"))  # type: ignore
        cancellation_token = cl.user_session.get("cancellation_token")  # type: ignore

        # Clear computation cancellation event at the start of each new message
        # This ensures that previous stop button clicks don't affect new computations
        computation_event = cl.user_session.get("computation_cancellation_event")  # type: ignore
        if computation_event and computation_event.is_set():
            computation_event.clear()
            print("Cleared computation cancellation event for new message")

        streaming_response: cl.Message | None = None

        # Inject session context into agent's system message (memory layer)
        session_state = cl.user_session.get("session_state")  # type: ignore
        if session_state:
            context_block = session_state.generate_context_block()
            if context_block:
                base_system_message = cl.user_session.get("base_system_message")  # type: ignore
                scientist_wrapper = cl.user_session.get("scientist_agent_wrapper")  # type: ignore
                if base_system_message and scientist_wrapper:
                    dynamic_message = AgentFactory.build_dynamic_system_message(
                        base_system_message, context_block
                    )
                    # Update the agent's system message
                    scientist_wrapper.update_system_message(dynamic_message)
                    scientist = scientist_wrapper.get_agent()

                    # Rebuild team with updated agent
                    termination = TextMentionTermination("?", sources=["Scientist"])
                    team = RoundRobinGroupChat(
                        [scientist],
                        max_turns=10,
                        termination_condition=termination
                    )
                    cl.user_session.set("team", team)

        # Rate limiting for OpenRouter is now handled per-request inside
        # RateLimitedOpenAIChatCompletionClient (see model_factory.py)

        # Stream responses from the group chat.
        async for msg in team.run_stream(
            task=[TextMessage(content=message.content, source="user")],
            cancellation_token=cancellation_token,
        ):
            if isinstance(msg, ModelClientStreamingChunkEvent):
                # Stream model client responses token-by-token.
                if streaming_response is None:
                    streaming_response = cl.Message(content="", author=msg.source)
                await streaming_response.stream_token(msg.content)
            elif streaming_response is not None:
                # Finish streaming and send the complete message.
                await streaming_response.send()
                streaming_response = None
            elif isinstance(msg, TaskResult):
                # If task termination is reached, send a final message.
                final_message = "Task terminated."
                if msg.stop_reason:
                    final_message += msg.stop_reason
                # await cl.Message(content=final_message).send()
            else:
                # Ignore other message types.
                pass
    except Exception as e:
        import datetime
        import traceback
        error_str = str(e)
        print(f"[{datetime.datetime.now()}] ERROR in chat handler: {e}")
        traceback.print_exc()

        # Check for Ollama JSON parsing errors (common with smaller models)
        # Error format: "error parsing tool call: raw='{""}', err=invalid character '}'"
        if ("invalid character" in error_str.lower() and "tool" in error_str.lower()) or \
           ("error parsing" in error_str.lower() and "tool call" in error_str.lower()):
            current_model = cl.user_session.get("current_model")  # type: ignore
            await cl.Message(
                content=(
                    f"⚠️ **Tool Call Error**\n\n"
                    f"The model `{current_model}` generated an invalid tool call response. "
                    f"This commonly happens with smaller Ollama models on complex tool calls.\n\n"
                    f"**Suggestions:**\n"
                    f"1. Switch to a larger Ollama model (`gpt-oss:20b` recommended)\n"
                    f"2. Use a cloud model (OpenRouter free tier or OpenAI)\n"
                    f"3. Try rephrasing your request more simply\n\n"
                    f"You can change models in the ⚙️ Settings panel."
                )
            ).send()
        # Check for Ollama out-of-memory error (model too large for available RAM)
        elif "requires more system memory" in error_str.lower():
            import re
            current_model = cl.user_session.get("current_model")  # type: ignore
            # Extract memory values from error: "requires more system memory (9.9 GiB) than is available (7.3 GiB)"
            required = re.search(r'requires more system memory \(([\d.]+) GiB\)', error_str)
            available = re.search(r'than is available \(([\d.]+) GiB\)', error_str)
            req_str = f"**{required.group(1)} GB**" if required else "more memory"
            avail_str = f"**{available.group(1)} GB**" if available else "what's available"
            await cl.Message(
                content=(
                    f"⚠️ **Not Enough Memory**\n\n"
                    f"The model `{current_model}` needs {req_str} of RAM, "
                    f"but only {avail_str} is currently free.\n\n"
                    f"**What to do:**\n"
                    f"1. Close other running applications to free up memory\n"
                    f"2. Use `gpt-oss:120b-cloud` instead (runs remotely, no local memory needed)\n"
                    f"3. Use OpenRouter free models (cloud-based, no local memory needed)\n\n"
                    f"You can change models in the ⚙️ Settings panel."
                )
            ).send()
        # Check for Ollama 401 unauthorized — device needs authorization
        elif "unauthorized" in error_str.lower() and "401" in error_str:
            current_model = cl.user_session.get("current_model")  # type: ignore
            # Get connect URL from ollama login
            login_url = ""
            try:
                import subprocess
                login_result = subprocess.run(
                    ["ollama", "login"], capture_output=True, text=True, timeout=10
                )
                for line in (login_result.stdout + login_result.stderr).split("\n"):
                    if "https://ollama.com/connect" in line:
                        login_url = line.strip()
                        break
            except Exception:
                pass

            if login_url:
                await cl.Message(
                    content=(
                        f"🔑 **Device Authorization Required**\n\n"
                        f"The cloud model `{current_model}` needs this device to be authorized.\n\n"
                        f"**Steps:**\n"
                        f"1. [Click here to authorize this device]({login_url})\n"
                        f"2. Log in to your Ollama account\n"
                        f"3. Click **\"Connect\"** on the page\n"
                        f"4. Try your message again\n\n"
                        f"This only needs to be done once."
                    )
                ).send()
            else:
                await cl.Message(
                    content=(
                        f"🔑 **Device Authorization Required**\n\n"
                        f"The cloud model `{current_model}` returned 401 Unauthorized.\n\n"
                        f"Please run `ollama login` in the terminal and authorize this device at the provided URL."
                    )
                ).send()
        # Check for OpenRouter provider errors — show friendly message instead of traceback
        elif _is_provider_error(error_str):
            current_model = cl.user_session.get("current_model")  # type: ignore
            reason = _get_provider_error_reason(error_str)
            await cl.Message(
                content=(
                    f"⚠️ **Model Temporarily Unavailable**\n\n"
                    f"The model `{current_model}` is currently experiencing issues ({reason}).\n\n"
                    f"**What to do:**\n"
                    f"1. Try again in a few minutes\n"
                    f"2. Switch to a different model in ⚙️ Settings\n"
                )
            ).send()
        else:
            await cl.Message(content=f"❌ **Error occurred**: {str(e)}\n\nPlease check the terminal logs for details.").send()


@cl.on_stop  # type: ignore
async def on_stop() -> None:
    """
    Handle stop button - cancel long-running computations only.

    This handler signals ongoing computations (like elastic tensor calculations)
    to stop gracefully at the next checkpoint. It does NOT cancel the agent itself
    to avoid putting the team in an inconsistent state (per AutoGen docs).

    The computation cancellation is cooperative - tools check the event periodically
    and raise ComputationCancelledException when set, then return a proper result
    indicating cancellation.

    **Design Choice**: Stop button only works during tool execution, not during
    agent streaming. This preserves conversational history and prevents team state
    corruption. Agent streaming is fast anyway, and max_turns=10 guardrail prevents
    infinite loops.
    """
    # DON'T cancel AutoGen agent task - causes inconsistent state per AutoGen docs
    # Let the agent finish its current turn to maintain clean team state

    # Signal ongoing computations to stop (cooperative cancellation)
    computation_event = cl.user_session.get("computation_cancellation_event")  # type: ignore
    if computation_event:
        computation_event.set()
        print("User clicked stop button - signalling computation cancellation")


@cl.on_chat_end  # type: ignore
def on_chat_end():
    """Handle session end (user disconnect or session termination)."""
    import datetime
    print(f"[{datetime.datetime.now()}] SESSION ENDED - User disconnected or session terminated")

    # Trigger cancellation for any running computations
    computation_event = cl.user_session.get("computation_cancellation_event")  # type: ignore
    if computation_event:
        computation_event.set()
        print("Signalling computation cancellation on session end")

    # Session cleanup complete (API key remains in .env file for future sessions)
