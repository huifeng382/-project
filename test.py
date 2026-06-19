import pandas as pd
df = pd.read_parquet("data/batch_02/timing_arcs.parquet")
print(df[df['DELAY'] <= 0])