import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, SeparableConv2D
from tensorflow.keras.optimizers import Adam, RMSprop

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class ModelCreation():

	def create_CNN(self, num_classes, seed=42):
		# Configurar cuDNN
		tf.keras.backend.set_floatx('float32')

		# Criar modelo com configurações otimizadas
		inputs = Input(shape=(224, 224, 3))
		
		# Inicializar MobileNetV3Small com configurações otimizadas
		base_model = MobileNetV3Small(
			include_top=False,
			weights='imagenet',
			input_tensor=inputs,
			minimalistic=True,  # Usar versão mais leve
		)

		# Log de verificação somente para o primeiro cliente
		# if (int(os.environ['CLIENT_ID']) == 0):
		# 	base_model.summary()

		base_model.trainable = False
		x = base_model.output
		
		# Global pooling and dense layers
		x = GlobalAveragePooling2D()(x)
		x = Dense(
			32,
			activation="relu",
			kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
			kernel_regularizer=tf.keras.regularizers.l2(1e-4)
		)(x)

		x = BatchNormalization(momentum=0.95)(x)
		x = Dropout(0.25)(x)

		outputs = Dense(num_classes, activation="softmax", name="classification")(x)
		deep_cnn = Model(inputs=inputs, outputs=outputs)

		optimizer = Adam(learning_rate=0.0001)

		deep_cnn.compile(
			optimizer=optimizer,
			loss="sparse_categorical_crossentropy",
			metrics=["accuracy"],
			# Adicionar jittering para melhor estabilidade no treinamento federado
        	jit_compile=True
		)

		# Log de verificação somente para o primeiro cliente
		# if (int(os.environ['CLIENT_ID']) == 0):
		# 	deep_cnn.summary()

		return deep_cnn