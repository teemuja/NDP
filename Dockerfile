FROM python:3.9

RUN apt-get update
RUN apt-get install -y libgdal-dev
RUN pip install GDAL==3.2.2.1

WORKDIR app/

COPY ./app .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

EXPOSE 8501