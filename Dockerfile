#FROM ufoym/deepo:latest
FROM nvcr.io/nvidia/pytorch:20.09-py3
RUN python -m pip install flask
ADD . /app
WORKDIR /app

EXPOSE 6008
ENTRYPOINT ["sh", "-c", "python control/pressure_control_service.py"]
CMD ["--cuda", "2"]
