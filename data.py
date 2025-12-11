import pandas as pd
import numpy as np
import os

#Burada verileri çekip düzenleyip modelde kullanılabilir hale getiriyorum.

# CSV dosyasının mutlak yolunu hesapla
_current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_current_dir, "housing.csv"))

dataForClustering = df[["longitude","latitude","median_income"]]

dataForRegression = df[["longitude","latitude","median_income","median_house_value"]]

dataForClassification = df[["longitude","latitude","median_income","median_house_value","ocean_proximity"]]