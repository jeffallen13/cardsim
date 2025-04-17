![CardSim](images/banner.png)
---

CardSim: A Payment Card Transaction Simulator
=======================================================

CardSim is a flexible, scalable Bayesian simulator for payment card fraud detection research. [Allen (2025)](https://www.federalreserve.gov/econres/feds/cardsim-a-bayesian-simulator-for-payment-card-fraud-detection-research.htm) presents the methodology in detail. 

## Setup

### Dependencies

`requirments.txt` lists the core dependencies and some optional installs 
depending on how you want to use the package. Installing `pandas` and `scipy`
should take care of all core dependencies, including `numpy`, which is used 
extensively.

To install all required dependencies, run:

```shell
pip install -r requirements.txt
```

### Installing CardSim from Source

You can install `cardsim` directly from GitHub.

```shell
pip install git+https://github.com/jeffallen13/cardsim.git
```

### Local Installation

Alternatively, you can install it locally. After cloning the repository, if you 
want to edit the source code, build the package in *editable* mode from within 
the project directory:

```shell
pip install -e .
```

If you prefer to use `cardsim` in another project without editing the source 
code, install it directly from the cloned path:

```shell
pip install /path/to/cardsim/
```

## Basic Usage

`vignette.py` shows how to use `cardsim`. The simulator generally works as 
follows: 

```python
from cardsim import Cardsim

# Instantiate the class
simulator = Cardsim()

# Run the simulator and store the transaction data
df = simulator.simulate()

# Export the transaction data
simulator.export_transaction_data(df)
```

The `Cardsim()` constructor supports numerous parameters, and `.simulate()` has 
additional options. Refer to the docstrings for details.

## Output Fields

The simulator output fields are defined below: 

| Field         | Definition |
|---------------|------------|
| date_time | The transaction datetime; transactions are sorted chronologically |
| payer_id | A unique identifier for the payer | 
| payee_id | A unique identifier for the payee | 
| amount | The payment amount | 
| credit_card | 1 if the payment is made using a credit card and 0 if using a debit card | 
| remote | 1 if the payment is made remotely and 0 if in-person | 
| distance | The Euclidean distance between the payer and payee | 
| fraud | 1 if the payment is fraudulent and 0 if it is legitimate | 

The output also includes several fields related to `date_time`, such as the hour
of day and the date, as well as a `run_id`, which is constructed as: 
`S{seed}P{n_payers}D{n_days}`. For example, a `run_id` could be: S42P500D180. 

## Calibration Sources

The simulator is calibrated to several publicly available data sources. The two most important are: 

- [Federal Reserve Payments Study (FRPS)](https://www.federalreserve.gov/paymentsystems/fr-payments-study.htm): The simulator uses the 2024 FRPS Networks, 
Processors, and Issuers Payments Surveys (NPIPS), which include data from calendar year 2022 (Board of Governors, 2024). 
- [Survey and Diary of Consumer Payment Choice](https://www.atlantafed.org/banking-and-payments/consumer-payments/survey-and-diary-of-consumer-payment-choice): The 
package uses data from the 2022 and 2023 surveys (Foster, Green, and Stavins, 2023, 2024). The data are located in the `cardsim/dcpc` folder. The vignette shows how to 
load the files. Storing them locally speeds up the simulator. Otherwise, the simulator will attempt to fetch them from the Atlanta Fed's website.

## References

> Allen, Jeffrey S. (2025). “CardSim: A Bayesian Simulator for Payment Card Fraud Detection Research." Finance and Economics Discussion Series 2025-017. Washington: Board of Governors of the Federal Reserve System, https://doi.org/10.17016/FEDS.2025.017.

> Board of Governors of the Federal Reserve System (2024). "The Federal Reserve Payments Study: Cards and Alternative Payments, 2021 and 2022." Washington: Board of Governors, https://www.federalreserve.gov/paymentsystems/fr-payments-study.htm.

> Foster, Kevin, Claire Greene, and Joanna Stavins (2023). "2022 Survey and Diary of Consumer Payment Choice." Federal Reserve Bank of Atlanta, https://doi.org/10.29338/rdr2023-03.

> Foster, Kevin, Claire Greene, and Joanna Stavins (2024). "2023 Survey and Diary of Consumer Payment Choice." Federal Reserve Bank of Atlanta, https://doi.org/10.29338/rdr2024-01.

## Disclaimer

The views presented in this repository are solely those of the author and should not be interpreted as reflecting the views of the Federal Reserve Board or the Federal Reserve System. 
