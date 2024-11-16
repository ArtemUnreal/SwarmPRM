FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]