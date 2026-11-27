FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NUMBA_CACHE_DIR=/tmp/numba
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[all]" && useradd -m hunter && mkdir -p /work && chown hunter /work
USER hunter
WORKDIR /work
# warm the JIT cache so the first cycle starts fast
RUN atsa bench --strategies 50 --bars 600 --workers 1 > /dev/null || true
EXPOSE 8000
CMD ["atsa", "run", "--forever"]
