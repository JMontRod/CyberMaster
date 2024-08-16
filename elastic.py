import requests
import json

# URL de Elasticsearch
es_url = "https://192.168.1.76:9200"

# Índice de Elasticsearch
es_index = "books"

api_key = "Elastic_Key"

# Realiza la solicitud a la API de Elasticsearch
response = requests.get(f"{es_url}/{es_index}/_search", verify=False, headers={'Content-Type': 'application/json', 'Authorization': f'ApiKey {api_key}'})

# Comprueba si la solicitud fue exitosa
if response.status_code == 200:
    print("Conexión exitosa a Elasticsearch")
    data = json.loads(response.text)
    total_docs = data['hits']['total']['value']
    if total_docs > 0:
        last_doc = data['hits']['hits'][-1]
        print(f"El último documento en el índice '{es_index}' es: {last_doc}")
        
        # Guardar el último documento en un archivo JSON
        with open("last_log.json", "w") as f:
            json.dump(last_doc, f)
    else:
        print(f"No se encontraron documentos en el índice '{es_index}'.")
else:
    print(f"Error al conectar a Elasticsearch: {response.status_code}")