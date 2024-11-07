import kaggle
import pandas as pd

kaggle.api.authenticate()

# Download latest version
kaggle.api.dataset_download_files("caiquecassemiro/eletrobras-base-historica", path = ".", unzip =True)

df = pd.read_csv("ELET3.csv")

print(df.head())


# Metadados

kaggle.api.dataset_metadata("caiquecassemiro/eletrobras-base-historica", path = ".")