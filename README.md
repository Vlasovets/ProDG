[![PyPI license](https://img.shields.io/pypi/l/gglasso.svg)](https://pypi.python.org/pypi/gglasso/)

[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# ProDG: Prokaryotic Data Generator

ProDG is a Python library for generating synthetic data based on different probability distributions. It's specifically designed for prokaryotic data but can be used for any type of data.

## Installation

You can install ProDG using pip:

```bash
pip install prodg
```

## Usage

Here is  a basic example of how to use ProDG:

```python
from prodg import DataGenerator

# create a sample data where rows are bacterial species names and columns are sample names
df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=['Sample1', 'Sample2', 'Sample3', 'Sample4'])

# call generator instance
prodg = DataGenerator()

# Fit the models to the data
prodg.fit(df)

# Generate new data
synthetic_data = prodg.generate(df)

# Print the synthetic data
print(synthetic_data)
```

## License

ProDG is licensed under the MIT License.

