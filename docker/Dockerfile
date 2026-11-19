FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY pipeline ./pipeline
COPY configs ./configs
RUN pip install --no-cache-dir -e ".[dev,video]"
COPY tests ./tests

CMD ["pytest", "-q"]
