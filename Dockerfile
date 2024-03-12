FROM python:3.10-bookworm

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY prepearing_data prepearing_data
COPY api api
COPY raw_data raw_data

#CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
#updated to the new version
#note: check https://stackoverflow.com/questions/43884981/unable-to-connect-localhost-in-docker and check from inspect THEIPV4ADDRESS in your web browser

CMD uvicorn api.fast_v2:app --port $PORT
