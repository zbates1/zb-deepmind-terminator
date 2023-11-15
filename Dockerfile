FROM nvcr.io/nvidia/pytorch:22.07-py3

# Install git
RUN apt-get update && apt-get install -y git

WORKDIR /wdr


# Clone the specific repo
RUN git clone https://github.com/instadeepai/nucleotide-transformer.git ./

RUN rm -f huggingface.py
RUN rm -f requirements.txt
RUN rm -f Dockerfile


# Install the Python requirements
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy everything from the current directory into the image
COPY . .
