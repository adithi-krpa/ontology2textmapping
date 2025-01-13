import pandas as pd
import csv


file_path = '/Users/shrikrupa/Desktop/proj test/radiology.csv'
chunk_size = 10000

df_generator = pd.read_csv(file_path, chunksize=chunk_size, on_bad_lines='skip', quoting=csv.QUOTE_NONE)

try:
    first_chunk = next(df_generator)
    if not first_chunk.empty:
        print(first_chunk.shape)
        print(first_chunk.iloc[888]["note_id"])  # Display the first row from the first chunk
    else:
        print("First chunk is empty or skipped.")
    second_chunk = next(df_generator)
    if not second_chunk.empty:
        print("\nSecond chunk shape:", second_chunk.shape)
        print("First row of second chunk:")
        print(second_chunk.iloc[888]["note_id"])  # Display the first row from the second chunk
    else:
        print("Second chunk is empty or skipped.")
except StopIteration:
    print("No more data available.")
except Exception as e:
    print(f"Error occurred: {e}")

