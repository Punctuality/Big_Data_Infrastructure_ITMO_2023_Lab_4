from typing import Union
import kafka

import logging as log

import pandas as pd

def send_results_to_bus(producer, topic: str, dataframe: pd.DataFrame) -> None:
    for index, row in dataframe.iterrows():
        row = row.to_json().encode('utf-8')
        index = str(index).encode('utf-8')
        producer.send(topic, key=index, value=row)
    producer.flush()
    log.debug("Producer finished sending messages")

  
def read_results_to_dataframe(consumer, topic: str) -> Union[pd.DataFrame, None]:
    consumer.subscribe([topic])
    log.debug("Consumer subscribed to topic")
    data = []
    columns = None
    while True:
        messages = consumer.poll(timeout_ms=1000, max_records=100)
        if not messages:
            break
        for _, msgs in messages.items():
            log.info(f"Read messages {len(msgs)}")
            for message in msgs:
                index = int(message.key.decode('utf-8'))
                row = pd.read_json(message.value.decode('utf-8'), typ='series')
                # check if columns were set
                if not columns:
                    columns=row.index.to_list()
                    log.info(f"Received columns: {columns}")

                data.append((index, row))
    
    log.info("Finished reading messages")
    
    dataframe = None
    if columns:
        dataframe = pd.DataFrame.from_dict(dict(data), orient='index', columns=columns)
        
    return dataframe