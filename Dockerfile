#https://github.com/OSGeo/gdal/pkgs/container/gdal
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.4

# Install python3-pip and git
RUN apt-get update && \
    apt-get install -y python3-pip software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get -y install git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR app/

COPY ./app .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

EXPOSE 8501