[![PyPI license](https://img.shields.io/pypi/l/gglasso.svg)](https://pypi.python.org/pypi/gglasso/)

[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# ProDG: Prokaryotic Data Generator

ProDG is a Python library for generating synthetic data based on different probability distributions. It's specifically designed for prokaryotic data but can be used for any type of data.

## Installation

You can install ProDG using pip:

```bash
pip install ProDG
```

## Usage

Here is  a basic example of how to use ProDG:

```python
from ProDG import DataGenerator

# Create a DataFrame where rows are bacterial species names and columns are sample names
df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), index=['species' + str(i) for i in range(100)], columns=['samples' + str(i) for i in range(4)])

# Create an instance of the DataGenerator class
generator = DataGenerator()

# Fit the models to the data
generator.fit(df)

# Generate new data
synthetic_data = generator.generate(df)

# Print the synthetic data
print(synthetic_data)
```

## License

ProDG is licensed under the MIT License.
