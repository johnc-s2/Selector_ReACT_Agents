FROM python:3.10

WORKDIR /app
COPY selector_react_agent/smtp_server.py /app/smtp_server.py

# Install dependencies, including python-dotenv
RUN pip install python-dotenv

CMD ["python", "/app/smtp_server.py"]
