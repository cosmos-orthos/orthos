FROM rust:1.87-slim

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Create virtual environment
RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install maturin numpy pytest

# Copy project files
COPY Cargo.toml Cargo.lock* ./
COPY src/ ./src/
COPY python/ ./python/
COPY pyproject.toml ./
COPY README.md ./
COPY examples/ ./examples/
COPY benches/ ./benches/

# Build the library with optimizations
ENV RUSTFLAGS="-C target-cpu=native"
RUN maturin develop --release

# Default command runs the benchmark
CMD ["python", "examples/benchmark.py"]
