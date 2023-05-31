
import argparse
import configparser
import os
from result_processing import read_results_to_dataframe

import kafka

import logging as log

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

    log.debug("Connecting to Kafka broker")

    consumer = kafka.KafkaConsumer(bootstrap_servers=config['kafka']['bootstrap_servers'], auto_offset_reset='earliest')
    dataframe = read_results_to_dataframe(consumer, config['kafka']['result_topic'])
    
    if dataframe is not None:
        log.info(f"Received results (stats):\n{dataframe.describe()}")
        log.info(f"Total number of fakes (label == 1): {dataframe['label'].value_counts()[1]} out of {len(dataframe)}")
    else:
        log.info("No results received")

    log.info("Finished reading results")