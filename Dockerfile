FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user backend/ backend/
COPY --chown=user api/ api/
COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

ENV PYTHONPATH=/app
EXPOSE 7860

# Ensure instance directory is writable by the non-root user
RUN mkdir -p /app/instance && chown -R user:user /app

USER user
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "api.index:app"]
