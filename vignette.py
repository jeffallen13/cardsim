"""
Build a payment transaction dataset using cardsim
"""

from importlib import resources
from cardsim import Cardsim

dcpc_path = resources.files("cardsim.dcpc")

simulator = Cardsim(seed=212, dcpc_folder=dcpc_path)

df = simulator.simulate()

simulator.export_transaction_data(df, folder='data')

params = simulator.export_run_parameters(
    df, folder='data', file_name='cardsim_runs'
)