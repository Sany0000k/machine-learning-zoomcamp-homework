FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir pipenv

COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

COPY predict.py model.bin ./

EXPOSE 9696

CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]