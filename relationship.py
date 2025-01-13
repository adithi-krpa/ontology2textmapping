import pandas as pd
import csv  # To use csv.QUOTE_NONE

chunk_size = 100000
df = pd.read_csv('/Users/shrikrupa/Desktop/proj test/radiology.csv', chunksize=chunk_size, on_bad_lines='skip', quoting=csv.QUOTE_NONE)

for chunks in df:
    print(chunks.shape)

for chunk in df:
    print(chunk.iloc[0])  # Display the first row from the first chunk
    break
first_chunk = next(df)  # This grabs the first chunk from the generator

# Print the shape of the first chunk
print(first_chunk.shape)

# Display the first row from the first chunk
print(first_chunk.iloc[0])