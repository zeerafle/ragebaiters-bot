FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY *.py ./
ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "bot.py"]
