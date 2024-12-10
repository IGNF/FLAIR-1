# Use a lightweight base image with Conda installed
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Install g++ / gcc for rasterio
#ENV DEBIAN_FRONTEND=noninteractive
#RUN apt-get update && \
#    apt-get -y install build-essential

# Copy the Conda environment file and the project into the container
COPY environment.yml .

RUN conda update conda -y && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Install dependencies using Conda
RUN conda env create -f environment.yml && \
    conda clean -a

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "flair", "/bin/bash", "-c"]
# Copy the project into the container
COPY src ./src
COPY setup.py ./
COPY configs ./configs

# Install the package in editable mode
RUN pip install --no-cache-dir --ignore-installed -e .

# Start Fast-API app
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "flair", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
