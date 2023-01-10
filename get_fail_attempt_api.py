import pandas as pd
import requests
if __name__ == '__main__':
    headers={"Content-Type":'application/json',"X-Metabase-Session": "c4c1f504-c97a-4ef9-ba4b-7eec157ba070"}
    url = 'https://metabase.ninjavan.co/api/card'
    response = requests.get(url=url, headers=headers)
    response.json()

    
