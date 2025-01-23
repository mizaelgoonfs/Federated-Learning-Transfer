import flwr as fl
import tensorflow
import random
import numpy as np
import tensorflow as tf
import os
import time
import sys

from dataset_utils import ManageDatasets
from model_definition import ModelCreation

import warnings
warnings.simplefilter("ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

MIN_BATTERY = 0.10 # Mínimo de 10% de bateria

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class FedClient(fl.client.NumPyClient):

	def __init__(self, cid=0, n_clients=None, epochs=1, 
				 model_name             = 'None', 
				 client_selection       = False, 
				 solution_name          = 'None', 
				 aggregation_method     = 'None',
				 dataset                = '',
				 perc_of_clients        = 0,
				 transmittion_threshold = 0.2):

		self.cid          = int(cid)
		self.n_clients    = n_clients
		self.model_name   = model_name
		self.local_epochs = epochs

		self.local_model  = None
		self.train_ds     = None
		self.test_ds      = None
		self.stats        = None

		#resources
		self.battery               = random.randint(90, 100)
		self.cpu_cost              = 0.0
		self.transmittion_prob     = 1
		self.transmittion_threshold = transmittion_threshold

		#logs
		self.solution_name      = solution_name
		self.aggregation_method = aggregation_method
		self.dataset            = dataset

		self.client_selection = client_selection
		self.perc_of_clients  = perc_of_clients

		#params
		if self.aggregation_method == 'POC':
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.perc_of_clients}"

		elif self.aggregation_method == 'DEEV': 
			self.solution_name = f"{solution_name}-{aggregation_method}"

		elif self.aggregation_method == 'None':
			self.solution_name = f"{solution_name}-{aggregation_method}"

		self.train_ds, self.test_ds, self.stats = self.load_data(self.dataset, n_clients=self.n_clients)
		self.local_model            			= self.create_model()

	def load_data(self, dataset_name, n_clients):
		return ManageDatasets(cid=self.cid).select_dataset(dataset_name, n_clients)

	def create_model(self):
		if self.model_name == 'CNN':
			return ModelCreation().create_CNN(self.stats['n_classes'])
		else:
			print("O modelo definido não está disponível!")
	

	def get_parameters(self, config):
		return self.local_model.get_weights()

	def fit(self, parameters, config):
		selected_clients   = []
		trained_parameters = []
		selected           = 0
		has_battery        = False
		total_time         = -1

		if config['selected_clients'] != '':
			selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(' ')]
		
		print("\n\nCLIENT_SELECTION: ", self.client_selection)
		start_time = time.process_time()
		if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
			
			#check if client has some battery available for training
			if self.battery >= MIN_BATTERY:
				self.local_model.set_weights(parameters)
				has_battery        = True
				selected           = 1
				history            = self.local_model.fit(self.train_ds, epochs=self.local_epochs)
				trained_parameters = self.local_model.get_weights()
		
				total_time         = time.process_time() - start_time
				size_of_parameters = sum(map(sys.getsizeof, trained_parameters))
				avg_loss_train     = np.mean(history.history['loss'])
				avg_acc_train      = np.mean(history.history['accuracy'])
				

				filename = f"logs/{self.dataset}/{self.solution_name}/{self.model_name}/train_client_{self.cid}.csv"
				os.makedirs(os.path.dirname(filename), exist_ok=True)

				self.battery -= total_time * 0.02 # Consumo de 2% de bateria mais o esforço (cpu_time) gasto
				self.cpu_cost = total_time

			#check transmission probability
			last_prob              = self.transmittion_prob
			self.transmittion_prob = random.uniform(0, 1)

			if last_prob >= self.transmittion_threshold and has_battery:
				with open(filename, 'a') as log_train_file:
					log_train_file.write(f"{config['round']}, {self.cid}, {selected}, {total_time}, {size_of_parameters}, {avg_loss_train}, {avg_acc_train}\n")
				
				return self.build_response(trained_parameters, self.stats['train_samples'])

			#transmission or train failled
			else:
				filename = f"logs/{self.dataset}/{self.solution_name}/{self.model_name}/failures_{self.cid}.csv"
				os.makedirs(os.path.dirname(filename), exist_ok=True)

				with open(filename, 'a') as log_failure_file:
					log_failure_file.write(f"{config['round']}, {self.cid}, {last_prob}, {self.battery}\n")

				return self.build_response(parameters, self.stats['train_samples'])
		else:
			return self.build_response(parameters, self.stats['train_samples'])	


	def build_response(self, parameters, total_train_samples_by_client):
		return parameters, int(total_train_samples_by_client), {
			'cid': self.cid,
			'transmittion_prob': self.transmittion_prob,
			'cpu_cost': self.cpu_cost
		}


	def evaluate(self, parameters, config):
		self.local_model.set_weights(parameters)
		loss, accuracy     = self.local_model.evaluate(self.test_ds)
		size_of_parameters = sum(map(sys.getsizeof, parameters))

		filename = f"logs/{self.dataset}/{self.solution_name}/{self.model_name}/evaluate_client_{self.cid}.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)

		with open(filename, 'a') as log_evaluate_file:
			log_evaluate_file.write(f"{config['round']}, {self.cid}, {size_of_parameters}, {loss}, {accuracy}\n")

		evaluation_response = {
			"cid"               : self.cid,
			"accuracy"          : float(accuracy),
			"transmittion_prob" : self.transmittion_prob,
			"cpu_cost"          : self.cpu_cost,
			"battery"           : self.battery
		}

		return loss, int(self.stats['test_samples']), evaluation_response


def main():
	# Importe e configure o TensorFlow primeiro
	from tf_config import configure_tensorflow
	configure_tensorflow()

	print(f'TensorFlow version: {tf.__version__}')
	
	client =  FedClient(
					cid                    = int(os.environ['CLIENT_ID']), 
					n_clients              = int(os.environ['NUM_CLIENTS']), # variable defined only in the test environment, as it refers to the datasets directories
					model_name             = os.environ['MODEL'], 
					client_selection	   = os.environ['CLIENT_SELECTION'] != 'False',
					epochs                 = int(os.environ['LOCAL_EPOCHS']), 
					solution_name          = os.environ['SOLUTION_NAME'],
					aggregation_method     = os.environ['ALGORITHM'],
					dataset                = os.environ['DATASET'],
					perc_of_clients        = float(os.environ['POC']),
					transmittion_threshold = float(os.environ['TRANSMISSION_THRESHOLD']),
					)
	
	fl.client.start_client(
        server_address=os.environ['SERVER_IP'],
        client=client.to_client()
    )


if __name__ == '__main__':
	main()
