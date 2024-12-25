import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ctgan import CTGAN

diabetes_df = pd.read_csv("diabetes.csv")
ctgan = CTGAN()
ctgan.fit(diabetes_df)
synthetic_data = ctgan.sample(5000)

synthetic_data_df = pd.DataFrame(synthetic_data, columns=diabetes_df.columns)
synthetic_data_df.to_csv('synthetic_diabetes.csv', index=False)