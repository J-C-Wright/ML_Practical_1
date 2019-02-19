FROM jupyter/scipy-notebook

EXPOSE 8888

RUN pip install xgboost

COPY . /home/jovyan/