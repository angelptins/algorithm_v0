# Use official Miniconda base image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Install necessary system dependencies (if needed)
RUN apt-get update && \
    apt-get install -y \
    git \
    build-essential && \
    rm -rf /var/lib/apt/lists/*
    
# Clone repository with necessary files.
RUN git clone https://github.com/angelptins/algorithm_v0.git

# Create and activate a Conda environment
RUN conda create --name myenv python=3.9

# Activate environment by default in the shell
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]