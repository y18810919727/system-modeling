FROM ufoym/deepo:latest
RUN python -m pip install flask
ADD . /app
WORKDIR /app

EXPOSE 6008
ENTRYPOINT ["python", "control/pressure_control_service.py"]
CMD ["--cuda", "2"]
