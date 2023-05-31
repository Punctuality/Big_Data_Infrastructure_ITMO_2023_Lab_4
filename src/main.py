import argparse
import configparser
import json
import os

import train
import eval
from model import FakeNewsClassifier
import device_config
from result_processing import send_results_to_bus

import torch as t
import pandas as pd

import logging as log

from kafka import KafkaProducer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define config path')
    parser.add_argument('--config_path', type=str, help='path to INI config file', required=True)
    parser.add_argument('--log_level', type=str, help='log level', required=False, default='INFO')
    
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    log_level = os.environ.get('LOG_LEVEL')
    if not log_level:
        log_level = args.log_level
    log_level = log_level.upper()
    log.basicConfig(level=log_level, format='%(asctime)s %(levelname)s %(message)s')

    if config.has_section('system') and config.get('system', 'device'):
        device = t.device(config['system']['device'])
    else:
        device = device_config.optimal_device()

    log.debug(f"Workdir: {os.getcwd()}")
    log.debug(f"Device of choice: {device}")

    model_path = config['paths']['model_path']
    vocab_path = config['paths']['vocab_path']

    if os.path.exists(model_path) and os.path.exists(vocab_path):
        log.info("Model and vocab are present")
        vocab = t.load(vocab_path)

        embedding_dim = int(config['model']['embedding_dim'])
        hidden_dim = int(config['model']['hidden_dim'])
        num_layers = int(config['model']['n_layers'])
        dropout = float(config['model']['dropout'])

        model = FakeNewsClassifier(len(vocab), embedding_dim, hidden_dim, num_layers, dropout).to(device)
        model.load_state_dict(t.load(model_path))

    else:
        log.info("Model and vocab are NOT present")
        log.info("Starting training")
        model, data = train.train_baseline(config, device)
        log.info("Training finished")
        vocab = data.vocab

    log.info("Evaluating model")
    result_eval: pd.DataFrame = eval.eval_model_on_test(model, device, config, vocab)
    log.info("Evaluation finished.")

    log.info("Sending results to result bus (kafka)")

    client = KafkaProducer(bootstrap_servers=config['kafka']['bootstrap_servers'])

    send_results_to_bus(client, config['kafka']['result_topic'], result_eval)

    log.info("Results sent. Exiting.")