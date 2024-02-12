FROM python:3.10-bookworm

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY prepearing_data prepearing_data
COPY api api
COPY raw_data raw_data

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
