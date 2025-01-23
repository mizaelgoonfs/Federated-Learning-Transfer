# FROM tensorflow/tensorflow:latest-gpu

# ADD data ./data
# ADD client.py .


# ADD dataset_utils.py .
# ADD model_definition.py .

# RUN pip install -U pip
# RUN pip install numpy "pandas<2.0.0" flwr

# CMD ["python", "-u", "client.py"]

# Usa a imagem específica do TensorFlow com suporte a GPU na versão 2.10.0
FROM tensorflow/tensorflow:latest-gpu

# Adiciona o script Python ao diretório raiz do contêiner
# ADD transfer_learning.py .

# Definir a variável de ambiente TFDS_DATA_DIR
# ENV TFDS_DATA_DIR=/datasets

# Atualiza o pip e instala as dependências necessárias
RUN pip install -U pip && \
    pip install numpy "pandas<2.0.0" scipy tensorflow-datasets flwr

# Define o comando padrão a ser executado quando o contêiner iniciar
# CMD ["python", "-u", "transfer_learning.py"]