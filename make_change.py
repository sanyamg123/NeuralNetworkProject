import pandas as pd 
df = pd.read_csv("outputkaggle.csv")
ar = df.as_matrix()
print ar.shape