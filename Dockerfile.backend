FROM python:3.11.10

WORKDIR /usr/src/

COPY . .
RUN apt-get update && \
    apt-get install -y \
    python3-pip

RUN pip install -r requirements.txt

EXPOSE 8000 8080

CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]