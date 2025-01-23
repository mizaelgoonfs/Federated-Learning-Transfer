import os
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple
import random
import pandas as pd
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import tensorflow_datasets as tfds

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class ManageDatasets():

	def __init__(self, cid: int, batch_size: int = 32, seed: int = 42):
		"""
        Inicializa o particionador para um cliente específico.
        
        Args:
            client_id: ID do cliente atual
            batch_size: Tamanho do batch para treinamento
            seed: Semente para reprodutibilidade
        """
		self.client_id = cid
		self.batch_size = batch_size
		self.seed = seed
		np.random.seed(seed)
	
	def create_dirichlet_distribution(
    self, n_clients: int, n_classes: int, alpha: float = 1.0
	) -> np.ndarray:
		"""
		Cria distribuições Dirichlet para particionar dados entre clientes por classe.

		Args:
			n_clients: Número total de clientes
			n_classes: Número total de classes
			alpha: Parâmetro de concentração (menor = mais heterogêneo)

		Returns:
			Array 2D (n_classes x n_clients) com as proporções para cada cliente por classe
		"""

		# Criar uma distribuição para cada classe
		distributions = np.zeros((n_classes, n_clients))
		for i in range(n_classes):
			class_seed = self.seed + i
			np.random.seed(class_seed)
			distributions[i] = np.random.dirichlet(alpha * np.ones(n_clients))
		return distributions


	def preprocess_image(
		self, image: tf.Tensor, label: tf.Tensor
	) -> Tuple[tf.Tensor, tf.Tensor]:
		"""
		Aplica preprocessamento padrão nas imagens.

		Args:
			image: Imagem de entrada
			label: Rótulo correspondente

		Returns:
			Tupla com imagem processada e rótulo
		"""
		image = tf.image.resize(image, [224, 224])
		image = preprocess_input(image)
		return image, label


	def create_train_test_split(
		self, dataset: tf.data.Dataset, train_ratio: float = 0.8
	) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
		"""
		Divide o dataset em treino e teste, aplicando data augmentation quando necessário
		para garantir batches completos.
		
		Args:
			dataset: Dataset a ser dividido
			train_ratio: Proporção para conjunto de treino
		
		Returns:
			Datasets de treino e teste, e estatísticas
		"""
		# Converter para lista para divisão
		data_list = list(dataset.as_numpy_iterator())
		n_samples = len(data_list)
		
		# Calcular números ideais de amostras para batches completos
		n_train = int(n_samples * train_ratio)
		n_test = n_samples - n_train
		
		train_batches = np.ceil(n_train / self.batch_size)
		test_batches = np.ceil(n_test / self.batch_size)
		
		train_samples_needed = int(train_batches * self.batch_size)
		test_samples_needed = int(test_batches * self.batch_size)
		
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
			.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
			.batch(self.batch_size)
			.prefetch(tf.data.AUTOTUNE))
		
		test_ds = (test_ds
			.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
			.batch(self.batch_size)
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


	def partition_dataset(
		self, dataset: tf.data.Dataset, n_clients: int, alpha: float = 1.0
	) -> Tuple[tf.data.Dataset, Dict]:
		"""
		Particiona o dataset por classe apenas para o cliente atual, garantindo exclusividade
		de amostras entre clientes.
		
		Args:
			dataset: Dataset completo
			n_clients: Número total de clientes
			n_classes: Número total de classes
			alpha: Parâmetro de concentração Dirichlet
				
		Returns:
			Dataset do cliente e estatísticas
		"""
		# Converter para lista e detectar número de classes
		data_list = list(dataset.as_numpy_iterator())
		labels = [item[1] for item in data_list]
		n_classes = len(np.unique(labels))  # Detecta automaticamente o número de classes
		total_samples = len(data_list)
		
		# Separar dados por classe
		class_data = [[] for _ in range(n_classes)]
		for item in data_list:
			class_idx = item[1]
			class_data[class_idx].append(item)
		
		# Gerar distribuições Dirichlet para cada classe
		class_distributions = self.create_dirichlet_distribution(
			n_clients, n_classes, alpha
		)
		
		# Calcular número de amostras por cliente para cada classe
		class_assignments = []
		for class_idx in range(n_classes):
			n_class_samples = len(class_data[class_idx])
			client_samples = (class_distributions[class_idx] * n_class_samples).astype(int)
			
			# Ajustar para garantir que soma seja igual ao total de amostras
			remaining = n_class_samples - np.sum(client_samples)
			if remaining > 0:
				# Distribuir amostras restantes para os clientes com maiores proporções
				sorted_clients = np.argsort(class_distributions[class_idx])[::-1]
				for i in range(remaining):
					client_samples[sorted_clients[i]] += 1
			
			class_assignments.append(client_samples)
		
		# Coletar dados para este cliente
		client_data = []
		class_counts = np.zeros(n_classes, dtype=int)
		
		for class_idx in range(n_classes):
			# Número de amostras desta classe para este cliente
			n_client_samples = class_assignments[class_idx][self.client_id]
			class_counts[class_idx] = n_client_samples
			
			# Calcular índices iniciais para cada cliente nesta classe
			client_start_indices = np.zeros(n_clients, dtype=int)
			for i in range(1, n_clients):
				client_start_indices[i] = (
					client_start_indices[i-1] + class_assignments[class_idx][i-1]
				)
			
			# Embaralhar dados desta classe (com seed específica)
			class_seed = self.seed + class_idx
			np.random.seed(class_seed)
			class_indices = np.random.permutation(len(class_data[class_idx]))
			
			# Selecionar amostras exclusivas para este cliente
			start_idx = client_start_indices[self.client_id]
			client_indices = class_indices[start_idx:start_idx + n_client_samples]
			
			# Adicionar dados desta classe para este cliente
			client_class_data = [
				class_data[class_idx][i] for i in client_indices
			]
			client_data.extend(client_class_data)
		
		# Embaralhar dados finais do cliente
		np.random.seed(self.seed + self.client_id)
		np.random.shuffle(client_data)
		
		# Separar imagens e rótulos
		images = np.array([item[0] for item in client_data])
		labels = np.array([item[1] for item in client_data])
		
		# Criar dataset
		client_ds = tf.data.Dataset.from_tensor_slices((images, labels))
		
		# Estatísticas
		stats = {
			'total_samples': total_samples,
			'n_classes': n_classes,
			'client_samples': len(client_data),
			'client_proportion': len(client_data) / total_samples,
			'class_distribution': {
				f'class_{i}': {
					'samples': count,
					'proportion': count / len(client_data) if len(client_data) > 0 else 0
				} for i, count in enumerate(class_counts)
			}
		}
		
		return client_ds, stats


	def save_stats_to_file(self, stats, output_dir='stats'):
		# Criar nome do arquivo e armazenar no diretório
		filename = f'client_{self.client_id}_stats.csv'
		filepath = os.path.join(output_dir, filename)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		
		# Preparar dados formatados
		lines = [
			f"Cliente ID,{self.client_id}\n",
			",\n",  # linha em branco
			"Distribuição de Dados,\n",
			f"Total de amostras do cliente,{stats['client_samples']}\n",
			f"Proporção do dataset global,{stats['client_proportion']:.2%}\n",
			",\n",  # linha em branco
			"Distribuição por Classe,\n",
		]
		
		# Adicionar distribuição por classe
		for class_name, info in stats['class_distribution'].items():
			lines.append(
				f"{class_name},{info['samples']} amostras ({info['proportion']:.2%})\n"
			)
		
		# Adicionar informações de treino/teste
		additional_lines = [
			",\n",  # linha em branco
			"Divisão Treino/Teste,\n",
			f"Amostras originais,{stats['original_samples']}\n",
			f"Amostras de treino,{stats['train_samples']} ({stats['augmented_train_samples']} aumentadas)\n",
			f"Amostras de teste,{stats['test_samples']} ({stats['augmented_test_samples']} aumentadas)\n",
			f"Batches de treino,{stats['train_batches']}\n",
			f"Batches de teste,{stats['test_batches']}\n",
			",\n",  # linha em branco
			"Informações do Dataset Global,\n",
			f"Total de amostras do dataset global,{stats['total_samples']}\n",
			f"Número de classes,{stats['n_classes']}\n"
		]
		
		lines.extend(additional_lines)
		
		# Salvar no arquivo
		with open(filepath, 'w', encoding='utf-8') as f:
			for line in lines:
				f.write(line)
		
		print(f"\nEstatísticas salvas em: {filepath}")


	def load_and_partition_dataset_EUROSAT(
		self,
		n_clients: int,
		alpha: float = 1.0,
		train_ratio: float = 0.8,
	) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
		"""
		Carrega, particiona e divide o dataset em treino/teste.

		Args:
			n_clients: Número total de clientes
			alpha: Parâmetro de concentração Dirichlet
			train_ratio: Proporção para conjunto de treino

		Returns:
			Datasets de treino e teste, e estatísticas combinadas
		"""

		print(f"\n{'='*50}")
		print(f"Inicializando Cliente: {self.client_id}")
		print(f"{'='*50}")

		# Carregar dataset
		full_ds, ds_info = tfds.load(
			"eurosat/rgb",
			split="train",  # Carrega o dataset completo sem divisão
			as_supervised=True,
			with_info=True,
			download=True  # Garante que os dados sejam baixados e salvos localmente
		)

		# Particionar para este cliente
		client_ds, partition_stats = self.partition_dataset(
			dataset=full_ds, n_clients=n_clients, alpha=alpha
		)

		# Dividir em treino/teste
		train_ds, test_ds, split_stats = self.create_train_test_split(
			dataset=client_ds, train_ratio=train_ratio
		)

		# Combinar estatísticas
		stats = {**partition_stats, **split_stats}

		# Salvar estatísticas
		self.save_stats_to_file(stats)

		# # Imprimir resumo detalhado
		# print(f"\nEstatísticas do Cliente {self.client_id}:")
		# print(f"{'='*50}")
		# print(f"\nDistribuição de Dados:")
		# print(f"- Total de amostras do cliente {self.client_id}: {stats['client_samples']:,}")
		# print(f"- Proporção do dataset global: {stats['client_proportion']:.2%}")
		
		# print(f"\nDistribuição por Classe:")
		# for class_name, info in stats['class_distribution'].items():
		# 	print(f"- {class_name}: {info['samples']:,} amostras ({info['proportion']:.2%})")
		
		# print(f"\nDivisão Treino/Teste (Data Augmentation para Batches sempre completos):")
		# print(f"- Amostras originais: {stats['original_samples']:,}")
		# print(f"- Amostras de treino: {stats['train_samples']:,} ({stats['augmented_train_samples']} aumentadas)")
		# print(f"- Amostras de teste: {stats['test_samples']:,} ({stats['augmented_test_samples']} aumentadas)")
		# print(f"- Batches de treino: {stats['train_batches']}")
		# print(f"- Batches de teste: {stats['test_batches']}")
		
		# print(f"\nInformações do Dataset Global:")
		# print(f"- Total de amostras do dataset global: {stats['total_samples']:,}")
		# print(f"- Número de classes: {stats['n_classes']}")
		# print(f"{'='*50}\n")

		return train_ds, test_ds, stats

	def select_dataset(self, dataset_name, n_clients):

		if dataset_name == 'EUROSAT':
			return self.load_and_partition_dataset_EUROSAT(n_clients=n_clients)
		else:
			print("O Dataset definido não está disponível!")