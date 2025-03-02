import flwr as fl
import numpy as np
import math
import os
import time

from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log

# from early_stopping_federated import EarlyStoppingFederated

class FedServer(fl.server.strategy.FedAvg):

	def __init__(self, aggregation_method, fraction_fit,
			  	num_clients, 
				perc_of_clients=0,
				dataset='',
				solution_name='',
				model_name=''):
		
		self.aggregation_method = aggregation_method
		self.num_clients        = num_clients
		self.list_of_clients    = []
		self.list_of_accuracies = []
		self.selected_clients   = []

		self.average_accuracy   = 0

		# self.stop_training = False
		# self.early_stopping = EarlyStoppingFederated(patience=5, min_delta=0.001)

		#pareto
		self.clients_info = {}

		#logs
		self.dataset    = dataset
		self.model_name = model_name

		#POC
		self.perc_of_clients  = perc_of_clients

		#params
		if self.aggregation_method == 'POC':
			self.solution_name = f"{solution_name}-{aggregation_method}-{self.perc_of_clients}"

		elif self.aggregation_method == 'DEEV': 
			self.solution_name = f"{solution_name}-{aggregation_method}"

		elif self.aggregation_method == 'None':
			self.solution_name = f"{solution_name}-{aggregation_method}"


		super().__init__(fraction_fit=fraction_fit, min_available_clients=num_clients, min_fit_clients=num_clients, min_evaluate_clients=num_clients)

	def configure_fit(self, server_round, parameters, client_manager):
		"""Configure the next round of training."""
		# Verifica se deve parar o treinamento antes de configurar o próximo round
		# if self.stop_training:
		# 	raise Exception("Early stopping ativado - Treinamento interrompido")

		if self.aggregation_method == 'POC':
			clients2select        = int(float(self.num_clients) * float(self.perc_of_clients))
			self.selected_clients = self.list_of_clients[:clients2select]

		elif self.aggregation_method == 'DEEV':
			self.selected_clients = self.select_clients_bellow_average()


		print(f'SELECTED CLIENTS: {self.selected_clients}')
		filename = f"logs/{self.dataset}/{self.solution_name}/{self.model_name}/selected_clients.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)

		with open(filename, 'a') as selected_clients_log_file:
			selected_clients_log_file.write(f"{server_round}, {self.selected_clients}\n")
		
		config = {
			"selected_clients" : ' '.join(self.selected_clients),
			"round"            : server_round
			}

		fit_ins = FitIns(parameters, config)

		# Sample clients
		sample_size, min_num_clients = self.num_fit_clients(
		    client_manager.num_available()
		)
		clients = client_manager.sample(
		    num_clients=sample_size, min_num_clients=min_num_clients
		)

		# Return client/config pairs
		return [(client, fit_ins) for client in clients]


	def aggregate_fit(self, server_round, results, failures):		
		weights_results = []

		for _, fit_res in results:
			client_id         = str(fit_res.metrics['cid'])
			transmittion_prob = float(fit_res.metrics['transmittion_prob'])
			cpu_cost          = float(fit_res.metrics['cpu_cost'])

			self.clients_info[client_id] = {}
			self.clients_info[client_id] = {'transmittion_prob' : transmittion_prob , 
											'cpu_cost': cpu_cost}

			if self.aggregation_method not in ['POC', 'DEEV'] or int(server_round) <= 1:
				weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

			else:
				if client_id in self.selected_clients:
					weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

		parameters_aggregated = fl.common.ndarrays_to_parameters(aggregate(weights_results))

		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}

		return parameters_aggregated, metrics_aggregated

	def configure_evaluate(self, server_round, parameters, client_manager):
		"""Configure the next round of evaluation."""
		# Do not configure federated evaluation if fraction eval is 0.
		if self.fraction_evaluate == 0.0:
			return []

		# Parameters and config
		config = {
			'round' : server_round
		}
		if self.on_evaluate_config_fn is not None:
			# Custom evaluation config function provided
			config = self.on_evaluate_config_fn(server_round)
		evaluate_ins = fl.common.EvaluateIns(parameters, config)

		# Sample clients
		sample_size, min_num_clients = self.num_evaluation_clients(
			client_manager.num_available()
		)
		clients = client_manager.sample(
			num_clients=sample_size, min_num_clients=min_num_clients
		)

		# Return client/config pairs
		return [(client, evaluate_ins) for client in clients]



	def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):

		local_list_clients      = []
		self.list_of_clients    = []
		self.list_of_accuracies = []
		accs                    = []

		
		for response in results:
			client_id       = response[1].metrics['cid']
			client_accuracy = float(response[1].metrics['accuracy'])
			client_trans    = float(response[1].metrics['transmittion_prob'])
			client_cputime  = float(response[1].metrics['cpu_cost'])
			client_battery  = float(response[1].metrics['battery'])

			filename = f"logs/{self.dataset}/{self.solution_name}/{self.model_name}/pareto.csv"
			os.makedirs(os.path.dirname(filename), exist_ok=True)

			with open(filename, 'a') as pareto_file:
				pareto_file.write(f"{server_round}, {client_id}, {client_accuracy}, {client_trans}, {client_cputime}, {client_battery}\n")

			accs.append(client_accuracy)

			local_list_clients.append((client_id, client_accuracy))

		local_list_clients.sort(key=lambda x: x[1])

		self.list_of_clients    = [str(client[0]) for client in local_list_clients]
		self.list_of_accuracies = [float(client[1]) for client in local_list_clients]

		accs.sort()
		self.average_accuracy   = np.mean(accs)
		print("\n\nMédia da ACURÁCIA: ", self.average_accuracy)

		# Weigh accuracy of each client by number of examples used
		accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
		examples   = [r.num_examples for _, r in results]

		# Ponto apropriado para implementar parada do processo (Interromper os rounds)

		# Aggregate and print custom metric
		accuracy_aggregated = sum(accuracies) / sum(examples)

		print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")

		# Aggregate loss
		loss_aggregated = weighted_loss_avg(
		    [
		        (evaluate_res.num_examples, evaluate_res.loss)
		        for _, evaluate_res in results
		    ]
		)

		# Aggregate custom metrics if aggregation fn was provided
		top5 = np.mean(accs[-5:])
		top1 = accs[-1]

		filename = f"logs/{self.dataset}/{self.solution_name}/{self.model_name}/server.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)

		with open(filename, 'a') as server_log_file:
			server_log_file.write(f"{time.time()}, {server_round}, {accuracy_aggregated}, {top5}, {top1}\n")

		metrics_aggregated = { 
			"accuracy"  : accuracy_aggregated,
			"top-5"     : top5,
			"top-1"     : top1
		}

		# if self.early_stopping(accuracy_aggregated):
		# 	print(f"\nEarly stopping ativado no round {server_round}")
		# 	print(f"Melhor acurácia alcançada: {self.early_stopping.best_accuracy}")
		# 	self.stop_training = True
	
		return loss_aggregated, metrics_aggregated

	def select_clients_bellow_average(self):

		selected_clients = []

		for idx_accuracy in range(len(self.list_of_accuracies)):

			if (self.list_of_accuracies[idx_accuracy] < self.average_accuracy):
				selected_clients.append(self.list_of_clients[idx_accuracy])

		return selected_clients


def main():
	start_time = time.time()
	
	server =  FedServer(
						 aggregation_method = os.environ['ALGORITHM'], 
					     fraction_fit       = 1, 
					     num_clients        = int(os.environ['NUM_CLIENTS']), 
					     perc_of_clients    = float(os.environ['POC']), 
					     solution_name      = os.environ['SOLUTION_NAME'],
					     dataset            = os.environ['DATASET'],
					     model_name         = os.environ['MODEL'])
	
	try:
		fl.server.start_server(
			server_address=os.environ['SERVER_IP'],
			config=fl.server.ServerConfig(num_rounds=int(os.environ['NUM_ROUNDS'])), # O número máximo de rounds
			strategy=server, 
		)

		end_time = time.time()
		elapsade_time = end_time - start_time

		print(f"\nTempo decorrido: {elapsade_time:.2f} segundos")
	except Exception as e:
		end_time = time.time()
		elapsade_time = end_time - start_time

		print(f"\nTempo decorrido: {elapsade_time:.2f} segundos")

		if "Early stopping ativado" in str(e):
			print("Treinamento finalizado por Early Stopping")
		else:
			raise e

if __name__ == '__main__':
	main()

