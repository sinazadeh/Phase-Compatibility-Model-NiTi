# \lambda_2
This repository contains a sample implementation of using CatBoost for building a surrogate model for transformation temperatures of Shape Memory Alloys (SMAs). The sample code (found in [main.py](main.py)) showcases how CatBoost can be used to model this data and make predictions with a high degree of accuracy. The accompanying [raw_data.csv](raw_data.csv) file contains raw data points that can be used to train and validate the model. This data can also serve as a reference for researchers looking to explore the use of CatBoost in this particular field.

# Getting Started
To run the sample code, you will need to have CatBoost installed in your environment. The code was tested using version 1.0.6 of CatBoost. If you do not have it installed, you can follow the instructions [here](https://catboost.ai/en/docs/concepts/installation) to install it. Other necessary packages are: [NumPy](https://numpy.org/install/), [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html), [CBFV](https://github.com/kaaiian/CBFV), and [scikit-learn](https://scikit-learn.org/stable/install.html).

# Usage
- Clone this repository
- Open the main.py file in your preferred Python environment
- Run the code to train and validate the model
# Citing
If you use this code or the accompanying data set, please cite the original paper. This will ensure proper recognition of the work that has gone into this repository and help further the research in this field.

> S.H. Zadeh, A. Behbahanian, J. Broucek, M. Fan, G. Vazquez, M. Noroozi, W. Trehern, X. Qian, I. Karaman, R. Arroyave, An interpretable boosting-based predictive model for transformation temperatures of shape memory alloys, Comput. Mater. Sci. 226 (2023) 112225. https://doi.org/10.1016/j.commatsci.2023.112225.
