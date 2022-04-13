FROM python:3.8.5

WORKDIR /src/app


RUN pip install flask
RUN pip install pandas
RUN pip install tensorflow
RUN pip install matplotlib
RUN pip install faiss-cpu

COPY . .

EXPOSE 15000

CMD ["python", "app.py"]