import polars as pl

# Load dataset using scan_csv
tm_convey = (
    pl.scan_csv('data.csv', separator=',', infer_schema_length=10000)
)

# Collect into DataFrame and compute descriptive statistics
summary = tm_convey.collect().describe()

# Save summary stats to CSV
summary.write_csv('summary.csv')

# Print to console for verification
if __name__ == '__main__':
    print(summary)
