import polars as pl

# Load dataset and compute descriptive statistics
summary = pl.read_csv('data.csv').describe()

# Save summary stats to CSV
summary.write_csv('summary.csv')

# Print to console for verification
if __name__ == '__main__':
    print(summary)
