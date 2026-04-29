# ============================================================
# OptiMat Alloys Docker Image
# Dual conda environment: optimat-alloys (ORB + MACE) + optimat-nequip (NequIP)
# Resolves e3nn version conflict: MACE needs 0.4.4, NequIP needs >=0.5.6
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="OptiMat Alloys Team"
LABEL description="AI-powered materials science with ORB, MACE, and NequIP potentials"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1. System dependencies
#    - libgl1-mesa-glx, libegl1-mesa, libxrender1: OVITO headless rendering
#    - xvfb: virtual framebuffer for headless OpenGL
#    - git: required by some pip packages (orb-models)
#    - build-essential, cmake: orb-models and sqsgenerator compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    build-essential cmake \
    libgl1-mesa-glx libgl1-mesa-dri libegl-mesa0 libgbm1 \
    libglib2.0-0 libegl1-mesa libxrender1 libxkbcommon0 libfontconfig1 libdbus-1-3 libopengl0 \
    xvfb tini zstd \
    && rm -rf /var/lib/apt/lists/*

# 1b. Install Ollama (LLM inference server for local + cloud models)
RUN curl -fsSL https://ollama.com/install.sh | sh

# 2. Install Miniforge (conda-forge-first, smaller than Miniconda)
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh \
    && $CONDA_DIR/bin/conda clean -afy
ENV PATH=$CONDA_DIR/bin:$PATH

# 3. Create optimat-nequip environment (smaller, changes less frequently)
RUN conda create -n optimat-nequip python=3.11 -y \
    && conda run -n optimat-nequip pip install --no-cache-dir \
        torch>=2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && conda run -n optimat-nequip pip install --no-cache-dir \
        ase>=3.24.0 nequip>=0.6.0 "e3nn>=0.5.6" numpy>=1.26.4 scipy>=1.15.1 \
    && conda clean -afy

# 4. Create optimat-alloys environment (main app, larger)
RUN conda create -n optimat-alloys python=3.11 -y \
    && conda run -n optimat-alloys pip install --no-cache-dir \
        torch>=2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && conda run -n optimat-alloys pip install --no-cache-dir \
        chainlit>=2.2.0 pyyaml \
        autogen-agentchat "autogen-ext[openai]" "autogen-ext[ollama]" \
        ase>=3.24.0 numpy>=1.26.4 scipy>=1.15.1 \
        dm-tree==0.1.8 tqdm>=4.66.5 \
        orb-models "e3nn==0.4.4" "mace-torch>=0.3.14" \
        ovito phonopy phono3py sqsgenerator \
        plotly pandas "mechelastic>=1.2.0" "pyvista>=0.43.0" \
        "reportlab>=4.0.0" "kaleido>=0.2.1" \
    && conda clean -afy

# 5. Set working directory
WORKDIR /app

# 6. Environment variables
ENV PATH=$CONDA_DIR/envs/optimat-alloys/bin:$PATH
ENV NEQUIP_ENV_PATH=$CONDA_DIR/envs/optimat-nequip/bin
ENV NEQUIP_CACHE_DIR=/app/cache/nequip
ENV TORCH_COMPILE_DISABLE=1
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# 7. Copy application code
COPY . /app/

# 8. Make entrypoint executable
RUN chmod +x /app/docker-entrypoint.sh

# 9. Expose Chainlit port
EXPOSE 8000

# 10. Persistent data volumes
VOLUME ["/app/structures", "/app/cache/nequip", "/root/.ollama"]

# 11. Entrypoint with tini for proper signal handling and zombie reaping
ENTRYPOINT ["tini", "--", "/app/docker-entrypoint.sh"]
CMD ["chainlit", "run", "run_chat.py", "--host", "0.0.0.0", "--port", "8000"]
