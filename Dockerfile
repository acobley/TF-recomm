FROM gcr.io/tensorflow/tensorflow:latest-gpu
RUN apt-get update && apt-get install -y \
   git \
   wget \
   emacs
RUN pip install Numpy
RUN pip install Pandas
RUN git clone -b MultiGPU https://github.com/acobley/TF-recomm.git #recom14
RUN ./TF-recomm/download_data.sh
EXPOSE 81
