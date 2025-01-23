# FROM tensorflow/tensorflow:latest-gpu

# ADD server.py .

# RUN pip install numpy pandas flwr

# CMD ["python", "-u", "server.py"]

# Usa a imagem específica do TensorFlow com suporte a GPU na versão 2.10.0
FROM tensorflow/tensorflow:latest-gpu

# Adiciona o script Python ao diretório raiz do contêiner
# ADD transfer_learning.py .

# Definir a variável de ambiente TFDS_DATA_DIR
# ENV TFDS_DATA_DIR=/datasets

# Atualiza o pip e instala as dependências necessárias
RUN pip install -U pip && \
    pip install numpy pandas flwr

# Define o comando padrão a ser executado quando o contêiner iniciar
# CMD ["python", "-u", "transfer_learning.py"]