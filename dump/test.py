import pandas as pd

x = pd.read_parquet('x_985772.parquet').to_numpy()
flat_x = pd.read_parquet('flat_x_985772.parquet').to_numpy()
unflat_x = pd.read_parquet('unflat_x_985772.parquet').to_numpy()

print(x[0,x[0]!=0][:10])

print(flat_x[:10])

print(unflat_x[0,x[0]!=0][:10])
print(unflat_x[0,unflat_x[0]!=0][:10])

