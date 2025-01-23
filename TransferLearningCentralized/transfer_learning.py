import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


from model_definition import ModelCreation

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def preprocess_image(image, label):
	"""Função de pré-processamento"""
	image = tf.image.resize(image, [224, 224])
	image = preprocess_input(image)
	return image, label

def create_train_test_split(
		dataset: tf.data.Dataset, train_ratio: float = 0.8, batch_size: int = 32, seed: int = 42
	):
		"""
		Divide o dataset em treino e teste, aplicando data augmentation quando necessário
		para garantir batches completos.
		
		Args:
			dataset: Dataset a ser dividido
			train_ratio: Proporção para conjunto de treino
			batch_size: Tamanho do batch
            seed: Semente para reprodutibilidade do embaralhamento
		Returns:
			Datasets de treino e teste, e estatísticas
		"""
		# Converter para lista para divisão
		data_list = list(dataset.as_numpy_iterator())
		n_samples = len(data_list)

		# Embaralhar os dados
		np.random.seed(seed)
		indices = np.arange(n_samples)
		np.random.shuffle(indices)
		data_list = [data_list[i] for i in indices]
		
		# Calcular números ideais de amostras para batches completos
		n_train = int(n_samples * train_ratio)
		n_test = n_samples - n_train
		
		train_batches = np.ceil(n_train / batch_size)
		test_batches = np.ceil(n_test / batch_size)
		
		train_samples_needed = int(train_batches * batch_size)
		test_samples_needed = int(test_batches * batch_size)
		
		# Dividir dados
		train_data = data_list[:n_train]
		test_data = data_list[n_train:]

		# Função de data augmentation
		def augment_image(image, label):
			# Aplicar transformações aleatórias
			image = tf.image.random_flip_left_right(image)
			image = tf.image.random_brightness(image, 0.2)
			image = tf.image.random_contrast(image, 0.8, 1.2)
			return image, label
		
		# Preencher batches incompletos com dados aumentados
		if len(train_data) < train_samples_needed:
			samples_to_add = train_samples_needed - len(train_data)
			# Selecionar amostras aleatórias para aumentar
			augment_indices = np.random.choice(len(train_data), samples_to_add)
			for idx in augment_indices:
				img, label = train_data[idx]
				aug_img, aug_label = augment_image(img, label)
				train_data.append((aug_img.numpy(), aug_label))
		
		if len(test_data) < test_samples_needed:
			samples_to_add = test_samples_needed - len(test_data)
			augment_indices = np.random.choice(len(test_data), samples_to_add)
			for idx in augment_indices:
				img, label = test_data[idx]
				aug_img, aug_label = augment_image(img, label)
				test_data.append((aug_img.numpy(), aug_label))
		
		# Separar imagens e rótulos para treino
		train_images = np.array([item[0] for item in train_data])
		train_labels = np.array([item[1] for item in train_data])
		
		# Separar imagens e rótulos para teste
		test_images = np.array([item[0] for item in test_data])
		test_labels = np.array([item[1] for item in test_data])
		
		# Criar datasets
		train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
		test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
		
		# Aplicar pipeline de preprocessamento
		train_ds = (train_ds
			.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
			.batch(batch_size)
			.prefetch(tf.data.AUTOTUNE))
		
		test_ds = (test_ds
			.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
			.batch(batch_size)
			.prefetch(tf.data.AUTOTUNE))
		
		# Estatísticas
		stats = {
			'original_samples': n_samples,
			'train_samples': len(train_data),
			'test_samples': len(test_data),
			'augmented_train_samples': len(train_data) - n_train,
        	'augmented_test_samples': len(test_data) - n_test,
			'train_batches': int(train_batches),
			'test_batches': int(test_batches)
		}
		
		return train_ds, test_ds, stats

def load_and_preprocess_data(train_ratio, batch_size):
	"""Carrega e aplica pré-processamento ao dataset EUROSAT."""
	# Carregue todo o dataset
	full_ds, ds_info = tfds.load(
		"eurosat/rgb",
		split="train",  # Carrega o dataset completo sem divisão
		as_supervised=True,
		with_info=True,
		download=True  # Garante que os dados sejam baixados e salvos localmente
	)
    
	train_ds, test_ds, stats = create_train_test_split(
			dataset=full_ds, train_ratio=train_ratio, batch_size=batch_size
	)

	print('\n', stats, '\n')

	return train_ds, test_ds
	
def train_and_evaluate(model, train_dataset, test_dataset, epochs):
	# Caminho onde o modelo será salvo
	diretorio_modelo = "models"
	nome_arquivo_modelo = "best_model.keras"

	# Verificar se o diretório existe, se não, criá-lo
	if not os.path.exists(diretorio_modelo):
		os.makedirs(diretorio_modelo)

	# Configurando o ModelCheckpoint
	checkpoint = ModelCheckpoint(
		filepath=os.path.join(diretorio_modelo, nome_arquivo_modelo),
		save_best_only=True,
		monitor="val_loss",
	)

	early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

	# # Callback para agendamento da taxa de aprendizado
	# lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1)

	history = model.fit(
		train_dataset,
		validation_data=test_dataset,
		epochs=epochs,
		callbacks=[checkpoint, early_stopping],
	)

	return history


def main():
	start_time = time.time()

	# Importe e configure o TensorFlow primeiro
	from tf_config import configure_tensorflow
	configure_tensorflow()

	print(f'TensorFlow version: {tf.__version__}')

	num_classes = 10
	train_ratio = 0.8 
	batch_size = 32
	epochs = 100

	# Carrega e pré-processa os dados
	train_dataset, valid_dataset = load_and_preprocess_data(train_ratio, batch_size)

	model = ModelCreation().create_CNN(num_classes=num_classes)

	history_data = train_and_evaluate(
		model,
		train_dataset,
		valid_dataset,
		epochs=epochs,
	)

	# Criando DataFrames para dados de treino e validação
	train_df = pd.DataFrame({"accuracy": history_data.history["accuracy"], "loss": history_data.history["loss"]})

	validation_df = pd.DataFrame(
		{
			"val_accuracy": history_data.history[
				"val_accuracy"
			],  # Colocar o resultado em uma lista para criar o DataFrame
			"val_loss": history_data.history["val_loss"],
		}
	)

	# Especificando os caminhos dos arquivos
	train_filename = "results/train_data.csv"
	validation_filename = "results/validation_data.csv"

	# Salvando os DataFrames em arquivos CSV
	train_df.to_csv(train_filename, index=False)
	validation_df.to_csv(validation_filename, index=False)

	print(f"Dados de treino salvos em {train_filename}")
	print(f"Dados de validação salvos em {validation_filename}")

	end_time = time.time()
	elapsade_time = end_time - start_time

	print(f"\nTempo decorrido: {elapsade_time:.2f} segundos")

if __name__ == "__main__":
	main()
