FROM nvcr.io/nvidia/pytorch:20.09-py3
RUN python -m pip install flask hydra-core
ADD . /app
WORKDIR /app
ENV algorithm cem
ENV cuda 2

EXPOSE 6010
ENTRYPOINT ["sh", "-c", "python control/control_service.py cuda=$cuda algorithm=$algorithm"]

