# -----------------------------------------------------------------------------#
# Cardsim: A Bayesian simulator for payment card fraud detection research
# Author: Jeff Allen
# -----------------------------------------------------------------------------#

import os
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np
from scipy.stats import lognorm, triang

class Cardsim:
    '''
    A class for simulating payment card transactions and fraud.

    Attributes
    ----------
    DEFAULT_DISTANCE_MODE_QUANTILE : dict
        The quantiles to find the modes in the triangular distribution
        for the distance of in-person, remote, and fraudulent payments. 
        The triangular distribution is applied to the payee indices
        after sorting them in ascending order by distance for each payer.
    DEFAULT_MARGINAL_TOD_WEIGHTS : dict
        Weights for each component of the marginal time of day distribution.
    DEFAULT_MARGINAL_TOD_WINDOWS : dict
        Start and end hours for peak windows in the marginal time of day
        distribution. 
    DEFAULT_CONDITIONAL_TOD_WEIGHTS : dict
        Weights for each component of the conditional time of day distribution.
    DEFAULT_CONDITIONAL_TOD_WINDOWS : dict
        Start and end hours for peak windows in the conditional time of day
        distribution. 
    '''

    DEFAULT_DISTANCE_MODE_QUANTILE = {
        'in_person': 0.01, 'remote': 0.5, 'fraud': 0.75
    }

    DEFAULT_MARGINAL_TOD_WEIGHTS = {
        'baseline': 0.4,
        'breakfast': 0.2,
        'lunch': 0.2,
        'dinner': 0.2
    }

    DEFAULT_MARGINAL_TOD_WINDOWS = {
        'breakfast': (7, 9),
        'lunch': (11, 13),
        'dinner': (18, 21)
    }

    DEFAULT_CONDITIONAL_TOD_WEIGHTS = {
        'baseline': 0.5,
        'night': 0.3,
        'morning': 0.2,
    }

    DEFAULT_CONDITIONAL_TOD_WINDOWS = {
        'night': (22, 24),
        'morning': (0, 5)
    }

    def __init__(self, 
                 log_level: str = 'INFO', 
                 seed: Optional[int] = None,
                 dcpc_start_year: int = 2022, 
                 dcpc_end_year: int = 2023,
                 dcpc_folder: Optional[str] = None,
                 txns_samples_m: int = 2500, 
                 txns_samples_n: int = 100,
                 value_samples_m: int = 5000, 
                 value_samples_n: int = 200, 
                 grid_size: int = 200, 
                 payer_payee_factor: int = 10,
                 debit_fraud_mult: float = 2.0, 
                 credit_fraud_mult: float = 1.45,
                 credit_card_marginal_p: float = 0.38, 
                 credit_card_conditional_p: float = 0.57,
                 remote_marginal_p: float = 0.36, 
                 remote_conditional_p: float = 0.63,
                 distance_mode_quantile=None,
                 marginal_tod_weights=None,
                 marginal_tod_windows=None,
                 conditional_tod_weights=None,
                 conditional_tod_windows=None,
                 tod_smoothing_param : Optional[float] = 0.5,
                 fraud_rate: float = 0.01,
                 lr_cap: Union[float, int] = 5,
                 fraud_flag_threshold: float = 0.01):
        '''
        Create a payment transaction simulator.

        Parameters
        ----------
        log_level: str, optional
            Threshold for the logger. The default is 'INFO'.
        seed : int, optional
            A seed for reproducibility. The default is None.
        dcpc_start_year : int, optional
            The first year of data to use from the Diary of Consumer Payment 
            Choice (DCPC) in simulating payer characteristics. The default is 
            2022. 
        dcpc_end_year : int, optional
            The last year of data to use from the Diary of Consumer Payment 
            Choice (DCPC) in simulating payer characteristics. The default is 
            2023. 
        dcpc_folder : str or None
            A folder where the DCPC data live. If None, data are sourced from
            FRB Atlanta website. The default is None. The system needs at least
            one year of DCPC data and files for the following DCPC levels: 
            day, ind, tran. An example naming structure is: 
            dcpc_2023_daylevel_public_xls.csv. Cardsim ships with two years 
            of data in cardsim/dcpc. The vignette shows how to access the 
            folder path. The simulator is faster when loading files locally.
        txns_samples_m : int, optional 
            The number of samples to draw when simulating average number of 
            daily payment transactions. Increasing the number smooths out the 
            curve. The default is 2500. 
        txns_samples_n : int, optional 
            The size of the samples when simulating average number of 
            daily payment transactions. Increasing this number decreases the 
            variance. Some variance is good. The default is 100.             
        value_samples_m : int, optional 
            The number of samples to draw when simulating average values. 
            Increasing the number smooths out the curve. The default is 5000.
        value_samples_n : int, optional 
            The size of the samples when simulating average values. Increasing
            this number decreases the variance. Some variance is good. The 
            default is 200.
        grid_size : int, optional
            The size (x, y) of the grid in which payers and payees reside.
            For example, grid_size=100 will create a 100 X 100 grid. The
            default is 200.
        payer_payee_factor : int, optional
            Payer to payees factor. Used as a divisor in calculating number of
            payees based on the number of payers. The default is 10. Derived
            from DCPC and Census Bureau data. 
        debit_fraud_mult: float, optional
            Multiplier for the average value of fraudulent debit card 
            transactions. The default is 2.0. Derived from FRPS.
        credit_fraud_mult: float, optional
            Multiplier for the average value of fraudulent credit card 
            transactions. The default is 1.45. Derived from FRPS.
        credit_card_marginal_p: float, optional
            Probability that a card transaction is made with a credit card.
            Debit card probability calculated as 1-credit_card_marginal_p.
            The default is 0.38. Derived from FRPS.
        credit_card_conditional_p: float, optional
            The conditional probability of p(credit card | fraud). Also used
            in calculating debit card conditional conditional probability.
            The default is 0.57. Derived from FRPS.
        remote_marginal_p: float, optional
            Probability that a card transaction is made remotely. In person
            probability calculated as 1-remote_marginal_p. The default is 0.36.
            Derived from FRPS.
        remote_conditional_p: float, optional
            The conditional probability of p(remote | fraud). Also used
            in calculating in person conditional probability. The default is 
            0.63. Derived from FRPS.
        distance_mode_quantile: dict or None, optional
            The quantiles to find the modes in the triangular distribution
            for the distance of in-person, remote, and fraudulent payments. 
            The triangular distribution is applied to the merchant indices
            after sorting them in ascending order by distance for each payer.
            Defaults to None and inherits from DEFAULT_DISTANCE_MODE_QUANTILE
        marginal_tod_weights: dict or None, optional
            Weights for each component of the marginal time of day distribution. 
            Defaults to None and inherits from DEFAULT_MARGINAL_TOD_WEIGHTS.
        marginal_tod_windows : dict or None, optional
            Start and end hours for peak windows in the marginal time of day
            distribution. Defaults to None and inherits from 
            DEFAULT_MARGINAL_TOD_WINDOWS.
        conditional_tod_weights : dict or None, optional
            Weights for each component of the conditional time of day 
            distribution. Defaults to None and inherits from 
            DEFAULT_CONDITIONAL_TOD_WEIGHTS.
        conditional_tod_windows : dict or None, optional
            Start and end hours for peak windows in the conditional time of day
            distribution. Defaults to None and inherits from 
            DEFAULT_CONDITIONAL_TOD_WINDOWS.
        tod_smoothing_param : float or None, optional
            A parameter between 0-1 to smooth out the hourly probability 
            mass function. The default is 0.5. With a parameter of 0.5, the 
            smoothed hourly fraud probabilities would be
            (marginal_pmf * 0.5) + (conditional_pmf * (1-0.5)). There is a big 
            difference in the likelihood ratios for normal and fraudulent 
            transactions during certain hours, causing the Bayesian prediction 
            to generate very few or no fraudulent transactions during those 
            times, which may not be realistic. The smoothing parameter mitigates 
            this behavior. 
        fraud_rate: float, optional
            The prior probability of fraud assumed by the simulator.
            The default is 0.01.
        lr_cap: float or int, optional
            A cap for the likelihood ratios. In some rare cases, the likelihood
            ratios are very large. This could occur, for example, if a very high 
            value is drawn that is much more likely to be from the fraud 
            distribution. A cap ensures that a high ratio doesn't overpower the 
            calculation. The default is 5.  
        fraud_flag_threshold: float, optional
            The percent threshold to use for the fraud flag odds. Threshold 
            labels the top n% of transactions in terms of fraud odds. The 
            default is 0.01.
        Returns
        -------
            A `Cardsim` object.

        '''
        # Configures logging level for the class
        self.logger = logging.getLogger(__name__ + '.Cardsim')
        if self.logger.handlers:
            self.logger.handlers.clear() # Clear any existing handlers
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.setLevel(log_level)
        # Seeds: vary for key simulation components, so data are not identical
        self.base_seed = seed
        self.sample_rng = np.random.default_rng(self.derive_seed(1))
        self.payer_rng = np.random.default_rng(self.derive_seed(2))
        self.payee_rng = np.random.default_rng(self.derive_seed(3))
        self.transaction_rng = np.random.default_rng(self.derive_seed(4))
        self.fraud_rng = np.random.default_rng(self.derive_seed(5))
        # World
        self.dcpc_start_year = dcpc_start_year
        self.dcpc_end_year = dcpc_end_year
        self.dcpc_folder = dcpc_folder
        self.grid_size = grid_size
        self.payer_payee_factor = payer_payee_factor
        self.txns_samples_m = txns_samples_m
        self.txns_samples_n = txns_samples_n
        self.value_samples_m = value_samples_m
        self.value_samples_n = value_samples_n
        self.debit_fraud_mult = debit_fraud_mult
        self.credit_fraud_mult = credit_fraud_mult
        # Simulator
        self.credit_card_marginal_p = credit_card_marginal_p
        self.credit_card_conditional_p = credit_card_conditional_p
        self.remote_marginal_p = remote_marginal_p
        self.remote_conditional_p = remote_conditional_p
        self.distance_mode_quantile = (
            self.DEFAULT_DISTANCE_MODE_QUANTILE.copy()
            if distance_mode_quantile is None
            else distance_mode_quantile
        )
        self.marginal_tod_weights = (
            self.DEFAULT_MARGINAL_TOD_WEIGHTS.copy()
            if marginal_tod_weights is None
            else marginal_tod_weights
        )
        self.marginal_tod_windows = (
            self.DEFAULT_MARGINAL_TOD_WINDOWS.copy()
            if marginal_tod_windows is None
            else marginal_tod_windows
        )
        self.conditional_tod_weights = (
            self.DEFAULT_CONDITIONAL_TOD_WEIGHTS.copy()
            if conditional_tod_weights is None
            else conditional_tod_weights
        )
        self.conditional_tod_windows = (
            self.DEFAULT_CONDITIONAL_TOD_WINDOWS.copy()
            if conditional_tod_windows is None
            else conditional_tod_windows
        )
        self.tod_smoothing_param = tod_smoothing_param
        # Fraud generation
        self.fraud_rate = fraud_rate
        self.lr_cap = lr_cap
        self.fraud_flag_threshold = fraud_flag_threshold
        # Allocate storage for key components
        self.card_txns = None
        self.card_txns_daily = None
        self.atxns_distributions = None
        self.avalue_distributions = None
        self.payers = None
        self.payees = None
        self.n_payers = None
        self.n_payees = None
        self.distances = None
        self.n_days = None
        self.transactions = None
        self.tod_pmf = None
        self.card_likelihood_ratio = None
        self.location_likelihood_ratio = None
        self.value_likelihood_ratio = None
        self.distance_likelihood_ratio = None
        self.tod_likelihood_ratio = None
        self.fraud_posterior_odds = None
        self.run_id = None
        # Run the DCPC process once per instantiation
        self.source_format_dcpc_data()
        # Performance testing
        self.world_runtime = None
        self.transactions_runtime = None
        self.simulator_runtime = None

    def derive_seed(self, seed_modifier: int):
        """
        Derive a seed for a core component of the simulator.

        Parameters
        ----------
        seed_modifier : int
            A number for incrementing the seed.

        Returns
        -------
        Int or None
            A modified seed number or None.
        """
        if self.base_seed is not None:
            derived_seed = self.base_seed + seed_modifier
            self.logger.debug(f'derived seed is: {derived_seed}')
            return derived_seed
        else:
            self.logger.debug('No seed modifier passed - returning None')
            return None

    @staticmethod
    def import_dcpc_data(collection: str = 'day', 
                         start_year: int = 2022, 
                         end_year: int = 2023,
                         folder: Optional[str] = None):
        '''
        Import diary of consumer payment choice data. 

        Parameters
        ----------
        collection : str, optional 
            The collection to import. Valid options are 'day', 'tran', and 
            'ind'. The default is 'day'. 
        start_year : int, optional
            The first year to import. The default is 2022. 
        end_year : int, optional
            The last year to import. The default is 2023. 
        folder : str or None
            A folder where the DCPC data live. If None, data are sourced from
            FRB Atlanta website. The default is None. 

        Returns
        -------
        Pandas dataframe 
            A dataframe of DCPC data. 

        '''
        valid_collections = ['day', 'tran', 'ind']

        if collection not in valid_collections:
            raise ValueError(
                f"Invalid collection. Choose one of: {', '.join(valid_collections)}"
            )
        
        if collection=='day':
            cols = ['id', 'date', 'diary_day', 'ind_weight', 'dow_weight']
            sort = ['id', 'diary_day']
        elif collection=='tran':
            cols = ['id', 'diary_day', 'tran', 'pi', 'amnt']
            sort = ['id', 'diary_day', 'tran', 'pi']
        else:
            cols = ['id', 'cc_adopt', 'dc_adopt']
            sort = ['id']
        
        dfs = []
        
        if folder is not None:
            # Convert to Path object for OS-agnostic sourcing
            if hasattr(folder, '_paths'):
                # For MultiplexedPath (conda) and similar
                folder_path = Path(folder._paths[0])
            else: 
                folder_path = Path(folder)
        
        for year in range(start_year, end_year + 1):
            if folder is not None:
                source = (
                    folder_path / f"dcpc_{year}_{collection}level_public_xls.csv"
                )
            else:
                source = f"https://www.atlantafed.org/-/media/documents/banking/consumer-payments/survey-diary-consumer-payment-choice/{year}/dcpc_{year}_{collection}level_public_xls.csv"
            df = pd.read_csv(source, usecols=cols).sort_values(by=sort)
            df['year'] = year
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)
    
    def source_format_dcpc_data(self):
        '''
        Source and format the Diary of Consumer Payment Choice (DCPC) data. 

        Returns
        -------
        None. Populates self.card_txns and self.card_txns_daily. 
        '''

        # Import data ------------------
        self.logger.info('Sourcing DCPC data')
        try: 
            indivs = Cardsim.import_dcpc_data(
                collection='ind', 
                start_year = self.dcpc_start_year,
                end_year = self.dcpc_end_year,
                folder = self.dcpc_folder
            )

            daily = Cardsim.import_dcpc_data(
                collection='day', 
                start_year = self.dcpc_start_year,
                end_year = self.dcpc_end_year,
                folder = self.dcpc_folder
            )

            transactions = Cardsim.import_dcpc_data(
                collection='tran', 
                start_year = self.dcpc_start_year,
                end_year = self.dcpc_end_year,
                folder = self.dcpc_folder
            )
            self.logger.info('Sourcing successful; formatting data')
        except Exception as e:
            self.logger.info(f'Failed to get DCPC data with error: {e}')
            self.logger.info('Simulation ending')
            return
        
        # Format individual data ------------------
        '''
        Create indicator for whether respondent has adopted debit or credit 
        card. The code below assigns 0 where both are NA. That is OK here 
        because we will ultimately only retain those who have indicated that 
        they adopted a credit or debit card. 
        '''
        indivs['cc_dc'] = np.where(
            (indivs['cc_adopt'] == 1) | (indivs['dc_adopt'] == 1), 1, 0
        )

        # Format daily data ----------------------
        '''
        Remove those that don't have an ind_weight. These are mostly connected
        to respondents from the extra California pool. 
        '''
        daily = daily[~daily['ind_weight'].isnull()]

        # Count the number of diary days for id-year combinations 
        diary_days = (
            daily.groupby(['id', 'year'])['diary_day'].count().reset_index()
        )

        # Identify those who did not participate all 4 days [days 0-3]
        missing_all_days = diary_days['id'][diary_days['diary_day'] < 4].values

        # Drop those who did not participate all 4 days 
        daily = (
            daily[~daily['id'].isin(missing_all_days)].reset_index(drop=True)
        )

        '''
        Drop those without dow_weight. This mostly corresponds to day 0 and
        those who were assigned to days in September and November, which was
        meant to smooth out issues from diary fatigue. 
        '''
        daily = daily[~daily['dow_weight'].isnull()]

        # Merge cardholders
        daily = daily.merge(
            indivs[['id', 'year', 'cc_dc']], how='left', on=['id', 'year']
        )

        # Drop those without a card 
        daily = daily[daily['cc_dc'] == 1]

        # Format transactions data ----------------------
        # Need to get the dow_weight on the transactions data 
        transactions = transactions.merge(
            daily, on=['id', 'diary_day', 'year'], how='left'
        )

        transactions = transactions[~transactions['dow_weight'].isnull()]

        transactions['amnt_w'] = (
            transactions['amnt'] * transactions['dow_weight']
        )

        card_txns = (
            transactions.loc[transactions['pi'].isin([3.0, 4.0])].copy()
        )

        card_txns['card_type'] = np.where(
            card_txns['pi'] == 3.0, 'Credit', 'Debit'
        )

        # Add number of card transactions to daily data ----------------
        '''
        First, count the number of card transactions. The recommended unit of
        analysis for transaction data is id-diary_day. We are adding year 
        because we have multiple years. 
        '''
        card_txns_count = (
            card_txns.groupby(['id', 'diary_day', 'year'])['tran'].count()
            .reset_index().rename(columns={'tran': 'txns'})
        )

        card_txns_daily = daily.merge(
            card_txns_count, how='left', on=['id', 'diary_day', 'year']
        )

        card_txns_daily['txns'] = np.where(
            card_txns_daily['txns'].isnull(), 0, card_txns_daily['txns']
        )

        card_txns_daily['txns_w'] = (
            card_txns_daily['txns'] * card_txns_daily['dow_weight']
        )

        # Finally, store data 
        self.card_txns = card_txns
        self.card_txns_daily = card_txns_daily
        
        self.logger.info('DCPC data sourcing and formatting complete')
    
    def sample_payments(self, pmnt_series, m, n):
        '''
        Draw m samples of size n of a payments series. Used in simulating 
        representative payment values and daily number of transactions. 

        Parameters
        ----------
        pmnt_series : Pandas series
            A series of transaction values or counts. 
        m : int
            The number of samples to generate.
        n : int
            The size of the samples. 

        Returns
        -------
        Numpy array
            A numpy array of samples of size m, n
        '''
        return self.sample_rng.choice(pmnt_series, size=(m, n), replace=True)

    @staticmethod
    def calculate_mad(samples, scaled=True):
        '''
        Calculate the median absolute deviation of the simulated transaction
        samples.

        Parameters
        ----------
        samples : Numpy array
            A 2-D numpy array of size (m, n), where m is the number of samples
            and n is the size of each sample.
        scaled : bool
            Whether to use a scaled mad. The default is True.

        Returns
        -------
        Numpy array
            A vector of mean absolute deviations of size m.

        '''

        median = np.median(samples, axis=1, keepdims=True)

        abs_diff = np.abs(samples - median)

        mad = np.median(abs_diff, axis=1)

        if scaled:
            return mad * 1.4826  # See Wiki
        else:
            return mad

    def generate_pmnt_distributions(self):
        '''
        Generate distributions of representative values for number of daily 
        payment transactions and value of payments. Uses the mean for number 
        of payment transactions. We only need a single parameter for number of 
        transactions because eventually we use a Poisson distribution to 
        sample number of daily transactions. We need two parameters for payment
        value because we eventually draw payment values from a Lognormal 
        distribution. We use the median and scaled median absolute deviation 
        for representative payment values. 

        Returns
        -------
        None. Populates self.atxns_distributions and self.avalue_distributions
        '''

        # Average number of cards transactions 
        txns_samples = self.sample_payments(
            self.card_txns_daily['txns_w'], 
            m=self.txns_samples_m, 
            n=self.txns_samples_n
        )

        self.atxns_distributions = np.mean(txns_samples, axis=1)
        
        # Average value of payments 
        dc_value_samples = self.sample_payments(
            self.card_txns['amnt_w'][self.card_txns['card_type'] == 'Debit'], 
            m=self.value_samples_m, n=self.value_samples_n
        )

        cc_value_samples = self.sample_payments(
            self.card_txns['amnt_w'][self.card_txns['card_type'] == 'Credit'], 
            m=self.value_samples_m, n=self.value_samples_n
        )

        self.avalue_distributions  = pd.DataFrame({
            'dc_means': np.mean(dc_value_samples, axis=1),
            'dc_stds': np.std(dc_value_samples, axis=1),
            'dc_medians': np.median(dc_value_samples, axis=1),
            'dc_mad': Cardsim.calculate_mad(dc_value_samples),
            'cc_means': np.mean(cc_value_samples, axis=1),
            'cc_stds': np.std(cc_value_samples, axis=1),
            'cc_medians': np.median(cc_value_samples, axis=1),
            'cc_mad': Cardsim.calculate_mad(cc_value_samples),
        })

    @staticmethod
    def calculate_tvalue_params(mean, sd, mu=True):
        '''
        Calculate lognormal parameters to feed into transaction value
        generator. 

        Formulas: 
            mu = ln(m^2 / sqrt(m^2 + sd^2))
            sigma = sqrt(ln(1 + (sd^2 / m^2))

        Parameters
        ----------
        mean : float
            The average payment value for a payer.
        sd : float
            The standard deviation of payment values for a payer. 
        mu : bool
            True calculates the mu parameter. False calculates the sigma 
            parameter. The default is True. 

        Returns
        -------
        Numpy array
            Returns a 1-D numpy array

        '''

        if mu:
            return np.log(mean**2 / np.sqrt(mean**2 + sd**2))
        else:
            return np.sqrt(np.log(1 + (sd**2 / mean**2)))

    def generate_payer_profiles(self, n_payers: int):
        """Generate payer profiles.

        Parameters
        ----------
        n_payers : int
            The number of payers to generate.
        """
        self.logger.info(f'Generating payer profiles for {n_payers} payers')
        df = pd.DataFrame({
            'payer_id': range(n_payers),
            'payer_x': self.payer_rng.integers(0, self.grid_size, n_payers),
            'payer_y': self.payer_rng.integers(0, self.grid_size, n_payers),
            'mean_frequency': self.payer_rng.choice(
                self.atxns_distributions, size=n_payers
            )
        })
        
        sampled_indices = self.payer_rng.choice(
            self.avalue_distributions.index, size=len(df), replace=True
        )

        df['debit_mean'] = (
            self.avalue_distributions.loc[sampled_indices, 'dc_medians'].values
        )

        df['debit_sd'] = (
            self.avalue_distributions.loc[sampled_indices, 'dc_mad'].values
        )

        df['credit_mean'] = (
            self.avalue_distributions.loc[sampled_indices, 'cc_medians'].values
        )

        df['credit_sd'] = (
            self.avalue_distributions.loc[sampled_indices, 'cc_mad'].values
        )

        df['debit_mean_fraud'] = df['debit_mean'] * self.debit_fraud_mult
        
        df['credit_mean_fraud'] = df['credit_mean'] * self.credit_fraud_mult

        # Prep vars for lognormal distribution
        df['debit_ln_mu'] = Cardsim.calculate_tvalue_params(
            df['debit_mean'], df['debit_sd'], mu=True
        )

        df['credit_ln_mu'] = Cardsim.calculate_tvalue_params(
            df['credit_mean'], df['credit_sd'], mu=True
        )

        df['debit_ln_sd'] = Cardsim.calculate_tvalue_params(
            df['debit_mean'], df['debit_sd'], mu=False
        )

        df['credit_ln_sd'] = Cardsim.calculate_tvalue_params(
            df['credit_mean'], df['credit_sd'], mu=False
        )

        df['debit_ln_mu_fraud'] = Cardsim.calculate_tvalue_params(
            df['debit_mean_fraud'], df['debit_sd'], mu=True
        )

        df['credit_ln_mu_fraud'] = Cardsim.calculate_tvalue_params(
            df['credit_mean_fraud'], df['credit_sd'], mu=True
        )

        self.n_payers = n_payers # Store in class

        self.payers = df

    def generate_payee_profiles(self):
        '''
        Generate payee profiles.

        Returns
        ----------
        None. Populates self.payees and self.n_payees
        '''
        n_payees = int(self.payers.shape[0] / self.payer_payee_factor)
        self.logger.info(f'Generating payee profiles for {n_payees} payees')
        self.payees = pd.DataFrame({
            'payee_id': range(n_payees),
            'payee_x': self.payee_rng.integers(0, self.grid_size, n_payees),
            'payee_y': self.payee_rng.integers(0, self.grid_size, n_payees)
        })
        self.n_payees = n_payees

    def calculate_distances(self):
        '''
        Calculate the distance matrix between payers and payees and related
        components.

        Methodology:
        - Transform payer vector of length n into matrix of shape (n, 1)
        - Subtract payee vector of length m from payer matrix
        - Obtain matrix of shape (n, m)
        - Each element (i, j) is the difference between payer i and payee j
        - Overlay Euclidean distance calc: sqrt((x1 - x2)^2 + (y1 - y2)^2)
        - Ultimately, each entry is distance between payer i and payee j
        - Finally, convert to a long data frame with fields: 
        - payer, payee, distance, payee_order 
        - Long data frame will be merged with transactions data frame 

        Returns
        ----------
        None. Populates self.distances
        '''

        payers = self.payers
        payees = self.payees

        distance_matrix = np.sqrt(
            (payers['payer_x'].values[:, None] - payees['payee_x'].values)**2 +
            (payers['payer_y'].values[:, None] - payees['payee_y'].values)**2
        ).astype(np.float32)

        df = pd.DataFrame(distance_matrix)

        df = df.reset_index().rename(columns={'index': 'payer_id'})

        df_melted = df.melt(id_vars=['payer_id'], 
                            var_name='payee_id', 
                            value_name='distance')

        # Convert payee_id to integer (it was initially a column name)
        df_melted['payee_id'] = df_melted['payee_id'].astype(int)

        df_melted = (
            df_melted.sort_values(['payer_id', 'distance'])
            .reset_index(drop=True)
        )

        # Add the order of payees by distance for each payer for later use
        df_melted['payee_order'] = df_melted.groupby('payer_id').cumcount()

        self.distances = df_melted

    def generate_baseline_transactions(self, n_days: int, 
                                       start_date: str) -> pd.DataFrame:
        """
        Generate the baseline transactions for the simulator. Produces a 
        dataframe with payer IDs and dates corresponding to each transaction.

        Parameters
        ----------
        n_days : int
            Number of days the simulator should run. 
        start_date : str
            Fictional start date for the simulator in the format YYYY-MM-DD.

        Returns
        -------
        pd.DataFrame
            A data frame of transactions, where the number of rows corresponds
            to the number of transactions. 
        """

        self.n_days = n_days # Store number of days
        
        dates_df = pd.DataFrame(
            {'day_index': np.arange(n_days),
             'date': pd.date_range(start_date, periods=n_days)}
        )

        payer_fields = ['payer_id', 'mean_frequency']

        # Cross-join so that each payer is associated with each date
        dates_payers = pd.merge(
            dates_df, self.payers[payer_fields], how='cross'
        )

        # The number of transactions in a day, drawn from Poisson
        dates_payers['n_txn'] = self.transaction_rng.poisson(
            dates_payers['mean_frequency']
        )

        # Only retain observations that have transactions
        dates_payers = (
            dates_payers[dates_payers['n_txn'] > 0].reset_index(drop=True)
        )

        '''
        Explode the dataframe based on the number of transactions. Resulting
        number of (non-unique) payer-date combinations should correspond to 
        n_txn.
        '''
        exploded_df = dates_payers.reindex(
            dates_payers.index.repeat(dates_payers['n_txn'])
        )

        fields = ['day_index', 'date', 'payer_id']

        return exploded_df[fields].copy().reset_index(drop=True)

    def calculate_cp_complement(self, p_x: np.ndarray, 
                                p_x_given_fraud: np.ndarray) -> np.ndarray:
        """Calculate the complement of the conditional probability for a given 
        feature, P(X | !F), using the law of total probability. 

        Parameters
        ----------
        p_x : np.ndarray
            Probability of X, P(X)
        p_x_given_fraud : np.ndarray
            Probability of X given fraud, P(X | F)

        Returns
        -------
        np.ndarray
            An array of conditional probability complements
        """
        
        p_x_given_not_fraud = (
            (p_x - (p_x_given_fraud * self.fraud_rate)) / (1 - self.fraud_rate)
        )
        
        return p_x_given_not_fraud

    def generate_payment_attribute(self, n_samples: int, 
                                   atype: str = 'credit_card') -> np.ndarray:
        """
        Generate a card type or location type payment attribute and likelihood
        ratio. Card and location type follow the same generation logic. 

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        atype : str, optional
            The type of attribute to derive. Current options are 'credit_card'
            and 'remote'. The default is 'credit_card'. 

        Returns
        -------
        np.ndarray
            A vector of 0/1 values corresponding to the dummy variable
            attribute. Also populates relevant likelihood ratio container. 
        """

        valid_atypes = ['credit_card', 'remote']

        if atype not in valid_atypes:
            raise ValueError(
                "'atype' must be one of: " + ', '.join(valid_atypes)
            )

        if atype == 'credit_card':
            mp = self.credit_card_marginal_p
            cp = self.credit_card_conditional_p
        else:
            mp = self.remote_marginal_p
            cp = self.remote_conditional_p

        pmnt_attribute = self.transaction_rng.choice(
            [1, 0], size=n_samples, p=[mp, 1-mp]
        )

        p_x = np.where(pmnt_attribute == 1, mp, 1-mp)

        p_x_given_fraud = np.where(pmnt_attribute == 1, cp, 1-cp)

        p_x_given_not_fraud = self.calculate_cp_complement(
            p_x, p_x_given_fraud
        )

        likelihood_ratio = p_x_given_fraud / p_x_given_not_fraud

        likelihood_ratio = np.minimum(likelihood_ratio, self.lr_cap)

        if atype == 'credit_card':
            self.card_likelihood_ratio = likelihood_ratio
        else:
            self.location_likelihood_ratio = likelihood_ratio

        return pmnt_attribute
    
    def generate_transaction_value(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate transaction values and likelihood ratios.

        Parameters
        ----------
        df : pd.DataFrame
            A data frame of baseline transactions produced by
            `generate_baseline_transactions()`. 

        Returns
        -------
        np.ndarray
            A vector of transaction values. 
        """

        if 'day_index' not in df.columns:
            raise ValueError("'df' should be baseline transactions")
        
        df = df.copy()

        # Merge the payment amount details
        amount_vars = ['payer_id', 
                       'debit_ln_mu', 'debit_ln_sd', 'debit_ln_mu_fraud',
                       'credit_ln_mu', 'credit_ln_sd', 'credit_ln_mu_fraud']

        df = pd.merge(
            df, self.payers[amount_vars], how='left', on='payer_id'
        )

        # Create a single column pulling debit and credit params, where relevant
        df['mu'] = np.where(
            df['credit_card'] == 1, 
            df['credit_ln_mu'], 
            df['debit_ln_mu']
        )

        df['sigma'] = np.where(
            df['credit_card'] == 1,
            df['credit_ln_sd'], 
            df['debit_ln_sd']
        )

        df['fraud_mu'] = np.where(
            df['credit_card'] == 1, 
            df['credit_ln_mu_fraud'], 
            df['debit_ln_mu_fraud']
        )

        transaction_value = self.transaction_rng.lognormal(
            df['mu'], df['sigma']
        )

        '''
        Likelihood ratio calculations. Approximating probability with densities 
        using PDF. 
        '''
        p_x = lognorm.pdf(
            transaction_value,
            s=df['sigma'],
            scale=np.exp(df['mu'])
        )

        p_x_given_fraud = lognorm.pdf(
            transaction_value,
            s=df['sigma'],
            scale=np.exp(df['fraud_mu'])
        )

        p_x_given_not_fraud = self.calculate_cp_complement(
            p_x, p_x_given_fraud
        )

        value_likelihood_ratio = p_x_given_fraud / p_x_given_not_fraud

        self.value_likelihood_ratio = np.minimum(
            value_likelihood_ratio, self.lr_cap
        )

        return transaction_value

    def generate_add_payee_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate the payee distances and add them to the transactions data
        frame. 

        Parameters
        ----------
        df : pd.DataFrame
            A data frame of transactions that has had the location type added.

        Returns
        -------
        pd.DataFrame
            The input data frame with the payee distance added. 
        """

        if 'remote' not in df.columns:
            raise ValueError("'df' needs a location type column")

        df = df.copy()

        # Set up min, max, and mode indices for triangular distributions
        min_index = 0
        max_index = self.n_payees - 1 # zero-based indexing
        inperson_mode = self.distance_mode_quantile['in_person'] * max_index
        remote_mode = self.distance_mode_quantile['remote'] * max_index
        fraud_mode = self.distance_mode_quantile['fraud'] * max_index

        # Select mode for triangular distribution based on location type
        mode_vector = np.where(
            df['remote'] == 1, remote_mode, inperson_mode
        )

        # Draw payee indices from triangular distribution
        drawn_index = self.transaction_rng.triangular(
            left=min_index, mode=mode_vector, right=max_index, size=len(df)
        )

        '''
        Round indices and ensure within bounds
        Give the variable the same name as the var that will be merged 
        '''
        df['payee_order'] = np.clip(
            np.round(drawn_index).astype(int), min_index, max_index
        )

        # Merge the payee IDs and distances
        df = df.merge(self.distances, how='left', on=['payer_id', 'payee_order'])

        '''
        Distance likelihood ratio

        Scipy expects: x (value), loc (left), scale (right - left), and c,
        which is (mode - loc) / scale. Because left is simply 0 in this case,
        c simplifies to mode / max_index, and scale is simply max_index. 
        '''
        p_x = triang.pdf(
            drawn_index, 
            c=mode_vector / max_index, 
            loc=min_index,
            scale=max_index
        )         

        p_x_given_fraud = triang.pdf(
            drawn_index, 
            c=fraud_mode / max_index, 
            loc=min_index,
            scale=max_index
        )

        p_x_given_not_fraud = self.calculate_cp_complement(
            p_x, p_x_given_fraud
        )

        '''
        Working directly with the draws should prevent divide by zero errors. 
        If errors ever emerge, options are: (1) shift scale up by 1, but this 
        would require adjusting the c calculation, (2) working with distances, 
        but that would make the lookup more complicated. 
        '''

        distance_likelihood_ratio = p_x_given_fraud / p_x_given_not_fraud

        self.distance_likelihood_ratio = np.minimum(
            distance_likelihood_ratio, self.lr_cap
        )

        return df.drop(columns='payee_order')
    
    @staticmethod
    def calculate_time_density(weights: dict, windows: dict, 
                               tri_peak: float) -> np.ndarray:
        """
        Calculate hourly time density for marginal or conditional distributions

        Parameters
        ----------
        weights : dict
            Weights for each component distribution
        windows : dict
            Start and end hours for each window
        tri_peak : float
            Location of the peak for the triangular distribution (as a fraction
            of the day, e.g., 0.5 is noon)

        Returns
        -------
        np.ndarray
            Hourly density values
        """

        hours = np.arange(24)
        density = np.zeros(24)

        # Baseline triangular distribution. Evaluate density at midpoints.
        # (1) Scale hours to [0,1]
        scaled_hours = (hours + 0.5) / 24.0
        tri_dist = triang(c=tri_peak, loc=0, scale=1)
        # (2) Scale triangular density to match window components 
        # Dividing by 24 ensures that this sums to 1 over the day
        density += weights['baseline'] * tri_dist.pdf(scaled_hours) / 24.0

        # Now add peak densities one-by-one
        for window_name, (start, end) in windows.items():
            window_density = np.zeros(24)
            window_width = end - start
            window_mask = (hours >= start) & (hours < end)
            # Uniform density is 1/width in windows and 0 elsewhere
            window_density[window_mask] = 1.0 / window_width
            density += weights[window_name] * window_density

        return density
    
    def generate_hourly_probabilities(self):
        """Generate marginal and conditional probabilities for each hour.

        Returns
        -------
        None. Populates self.tod_pmf.
        """

        hours = np.arange(24)

        marginal_density = Cardsim.calculate_time_density(
            weights=self.marginal_tod_weights,
            windows=self.marginal_tod_windows,
            tri_peak=0.5
        )

        conditional_density = Cardsim.calculate_time_density(
            weights=self.conditional_tod_weights,
            windows=self.conditional_tod_windows,
            tri_peak=0.5
        )

        # Normalize to get a PMF
        marginal_pmf = marginal_density / np.sum(marginal_density)
        conditional_pmf = conditional_density / np.sum(conditional_density)

        df = pd.DataFrame({
            'hour': hours, 
            'marginal_pmf': marginal_pmf, 
            'conditional_pmf': conditional_pmf 
        })

        if self.tod_smoothing_param is not None:
            df['conditional_pmf'] = (
                (df['marginal_pmf'] * self.tod_smoothing_param) + 
                (df['conditional_pmf'] * (1 - self.tod_smoothing_param))
            )

        self.tod_pmf = df

    def generate_transaction_time(self, n_samples: int, 
                                  return_seconds: bool = True) -> np.ndarray:
        '''
        Generate a vector of times for payment transactions.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate. 
        return_seconds : bool
            True returns the time in seconds. False returns the time in hours. 
            The default is True. 

        Returns
        -------
        np.ndarray
            A vector of transaction times.

        '''

        hours = np.arange(24)
        probs = self.tod_pmf['marginal_pmf'].to_numpy()
        
        # Generate hours using hourly PMF
        selected_hours = self.transaction_rng.choice(
            hours, size=n_samples, p=probs
        )

        # Select random seconds
        selected_seconds = self.transaction_rng.integers(
            0, 3600, size=n_samples
        )

        if return_seconds:
            return selected_hours * 3600 + selected_seconds
        else:
            return selected_hours

    def calculate_tod_likelihood_ratio(self, df: pd.DataFrame):
        """
        Calculate the time of day likelihood ratios by merging the hourly 
        PMFs. 

        Parameters
        ----------
        df : pd.DataFrame
            A data frame of transactions that has had time of day features
            added.

        Returns
        -------
        None
            Populates self.tod_likelihood_ratio with an array of time of day 
            likelihood ratios. 
        """

        if 'hour' not in df.columns:
            raise ValueError("'df' should contain time of day elements")
        
        df = df.copy()

        df = pd.merge(df, self.tod_pmf, how='left', on='hour')

        p_x = df['marginal_pmf'].to_numpy()

        p_x_given_fraud = df['conditional_pmf'].to_numpy()

        p_x_given_not_fraud = self.calculate_cp_complement(
            p_x, p_x_given_fraud
        )

        tod_likelihood_ratio = p_x_given_fraud / p_x_given_not_fraud

        self.tod_likelihood_ratio = np.minimum(
            tod_likelihood_ratio, self.lr_cap
        )

    def generate_fraud(self) -> np.ndarray:  
        """Generate the fraud flag by ranking posterior odds produced by Bayes'
        rule.

        Returns
        -------
        np.ndarray
            An array of binary values (the fraud flag).
        """

        prior_odds = self.fraud_rate / (1 - self.fraud_rate)

        likelihood_ratio = (
            self.card_likelihood_ratio * 
            self.location_likelihood_ratio * 
            self.value_likelihood_ratio *
            self.distance_likelihood_ratio *
            self.tod_likelihood_ratio
        )
        
        posterior_odds = prior_odds * likelihood_ratio

        self.fraud_posterior_odds = posterior_odds

        threshold_odds = np.percentile(
            posterior_odds, (1 - self.fraud_flag_threshold) * 100
        )

        fraud_flag = (posterior_odds >= threshold_odds).astype(int)
        
        return fraud_flag

    def simulate(self, n_payers: int = 10000, n_days: int = 365, 
                 start_date: str = '2023-01-01') -> pd.DataFrame:
        """Run the payment transaction simulator.

        Parameters
        ----------
        n_payers : int, optional
            The number of payers, by default 10000
        n_days : int, optional
            Number of days the simulator should run, by default 365
        start_date : str, optional
            Fictional start date for the simulator in the format YYYY-MM-DD, 
            by default '2023-01-01'

        Returns
        -------
        pd.DataFrame
            A data frame of payment transactions, features, and a fraud flag.
        """

        world_start = time.time()
        
        self.logger.info('Starting phase one: generating simulator world\n')

        try:
            self.generate_pmnt_distributions()
            self.logger.info('\nGenerated payment distributions\n')
        except Exception as e:
            self.logger.info(
                f'Failed to generate payment distributions with error: {e}'
            )
            return

        try:
            self.generate_payer_profiles(n_payers=n_payers)
            self.logger.info('Generated payer profiles\n')
        except Exception as e:
            self.logger.info(
                f'Failed to generate payer profiles with error: {e}'
            )
            return

        try:
            self.generate_payee_profiles()
            self.logger.info('Generated payee profiles\n')
        except Exception as e:
            self.logger.info(
                f'Failed to generate payee profiles with error: {e}'
            )
            return

        try:
            self.calculate_distances()
            self.logger.info('Calculated distance matrix\n')
        except Exception as e:
            self.logger.info(
                f'Failed to calculate distance matrix with error: {e}'
            )
            return

        world_end = time.time()

        self.world_runtime = world_end - world_start
        
        self.logger.info(
            f'Generated world in {round(self.world_runtime)} seconds\n'
        )

        self.logger.info(
            'Starting phase two: generating transactions within world\n'
        )

        tx_start = time.time()
        
        try:
            df = self.generate_baseline_transactions(
                n_days=n_days, start_date=start_date
            )
            self.logger.info(
                f'Generated baseline transactions\n'
            )
        except Exception as e:
            self.logger.info(
                f'Failed to generate baseline transactions with error: {e}'
            )
            return

        try:
            df['credit_card'] = self.generate_payment_attribute(
                n_samples = len(df), atype='credit_card'
            )
            self.logger.info(
                f'Generated card type attribute\n'
            )
        except Exception as e:
            self.logger.info(
                f'Failed to generate card type attribute with error: {e}'
            )
            return

        try:
            df['remote'] = self.generate_payment_attribute(
                n_samples = len(df), atype='remote'
            )
            self.logger.info(
                f'Generated location type attribute\n'
            )
        except Exception as e:
            self.logger.info(
                f'Failed to generate location type attribute with error: {e}'
            )
            return

        try:
            df['amount'] = self.generate_transaction_value(df)
            self.logger.info(
                f'Generated transaction amount\n'
            )
        except Exception as e:
            self.logger.info(
                f'Failed to generate transaction amount with error: {e}'
            )
            return

        try:
            df = self.generate_add_payee_distance(df)
            self.logger.info(
                'Generated and added payee distance'
            )
        except Exception as e:
            self.logger.info(
                f'Failed to generate payee distance with error: {e}'
            )
            return 

        try:
            self.generate_hourly_probabilities()
            df['time_seconds'] = self.generate_transaction_time(
                n_samples=len(df)
            )
            df['date_time'] = (
                df['date'] + pd.to_timedelta(df['time_seconds'], unit='s')
            )
            df['hour'] = df['date_time'].dt.hour
            self.calculate_tod_likelihood_ratio(df)            
            self.logger.info(
                'Generated transaction time\n'
            )
        except Exception as e:
            self.logger.info(
                f'Failed to generate transaction time with error: {e}'
            )
            return            
        
        try:
            df['fraud'] = self.generate_fraud()
            self.logger.info(
                f'Generated fraud flag\n'
            )
        except Exception as e:
            self.logger.info(
                f'Failed to generate fraud flag with error: {e}'
            )
            return

        self.run_id = f'S{self.base_seed}P{n_payers}D{n_days}'
        
        df['run_id'] = self.run_id 

        tx_end = time.time()
        self.transactions_runtime = tx_end-tx_start
        self.logger.info(
            f'Generated transactions in {round(self.transactions_runtime)} seconds\n'
        )
        
        self.simulator_runtime = self.world_runtime + self.transactions_runtime
        self.logger.info(
            f'Simulator completed in {round(self.simulator_runtime)} seconds\n'
        )

        vars = ['run_id', 'day_index', 'date', 'time_seconds', 'hour',  
                'date_time', 'payer_id', 'payee_id', 'credit_card', 'remote', 
                'distance', 'amount', 'fraud']
        
        return df[vars]

    # Convenience -------------------------------------------------------------

    def export_transaction_data(self, df: pd.DataFrame, folder: str, 
                                csv: bool = True, 
                                file_name: Optional[str] = None):
        """Export transaction data to a .csv or a .pkl.

        Parameters
        ----------
        df : pd.DataFrame
            The transaction data
        folder : str
            The name of the destination folder. 
        csv : bool, optional
            True exports as a csv. False exports as a serialized pkl.
            The default is True.
        file_name : str or None, optional
            The name of the file. The default is None.
        """

        if not os.path.exists(folder):
            os.makedirs(folder)

        if file_name is None:
            file_name = f'transaction-data-{self.run_id}'
        else:
            file_name = file_name
        
        path = folder + '/' + file_name
        self.logger.info(f'Saving data to {path}')
        if csv:
            path = path + '.csv'
            df.to_csv(path, index=False)
        else:
            path = path + '.pkl'
            df.to_pickle(path)

    def export_run_parameters(self, df: pd.DataFrame, folder: str, 
                              file_name: str, 
                              return_params: bool = True) -> pd.DataFrame:
        """Export relevant run parameters. 

        Parameters
        ----------
        df : pd.DataFrame
            The transaction data
        folder : str
            The destination folder
        file_name : str
            The name of the file
        return_params : bool, optional
            Whether to return the dataframe, by default True

        Returns
        -------
        pd.DataFrame
            A data frame of run parameters
        """

        df = pd.DataFrame({
            'run_id': [self.run_id],
            'n_payers': [self.n_payers],
            'n_payees': [self.n_payees],
            'n_days': [self.n_days],
            'n_obs': [len(df)],
            'fraud_rate': [self.fraud_rate],
            'runtime': [self.simulator_runtime]
        })

        path = f'{folder}/{file_name}.csv'

        if os.path.isfile(path):
            params = pd.read_csv(path)
            df = pd.concat([params, df], ignore_index=True)
            df = df.drop_duplicates(subset='run_id', keep='first')
        
        df.to_csv(path, index=False)

        if return_params:
            return df
