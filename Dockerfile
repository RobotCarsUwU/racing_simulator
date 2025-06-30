FROM ubuntu:18.04

# ----------- 1. Install system dependencies ----------
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.6 \
    python3.6-dev \
    wget \
    curl \
    git \
    build-essential \
    libhdf5-dev \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev

# ----------- 2. Set python3.6 as default ----------
RUN ln -sf /usr/bin/python3.6 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/pip/3.6/get-pip.py | python3.6

# ----------- 3. Copy your app ----------
WORKDIR /workspace
COPY . .

# ----------- 4. Install Python dependencies ----------
RUN python3.6 -m pip install --upgrade pip
RUN python3.6 -m pip install -r requirements.txt

# ----------- 5. Run ----------
CMD ["python3.6", "main.py"]

