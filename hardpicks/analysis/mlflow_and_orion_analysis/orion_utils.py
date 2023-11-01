import pandas as pd
from orion.client import get_experiment
from orion.core.utils.singleton import update_singletons


def get_orion_experiment_dataframe(orion_database_path: str, experiment_name: str) -> pd.DataFrame:
    """Extract all experimental data from Orion.

    CAREFUL! Orion creates a singleton "storage" object under the hood. The paradigm
    seems to be that there is one and only one database. If the "get_experiment" method
    is called naively twice in a row with different "storage" parameters with a different path,
    a new database is NOT read.

    To avoid this problem, the "update_singletons" method must be invoked to clean up the
    storage object and start clean with a new database.

    Arguments:
        orion_database_path: path to the orion database
        experiment_name: name of the experiment
    """
    update_singletons()  # defensively cleaning whatever singleton database might be lurking

    # Specify the database where the experiments are stored. We use a local PickleDB here.
    storage = dict(type="legacy", database=dict(type="pickleddb", host=orion_database_path))
    # Load the data for the specified experiment
    experiment = get_experiment(experiment_name, storage=storage)
    df = experiment.to_pandas()
    update_singletons()  # cleaning up after we are done

    return df


def get_orion_database_results(orion_database_path: str, experiment_name: str) -> pd.DataFrame:
    """Extract the jobs and their status from an orion database."""
    df = get_orion_experiment_dataframe(orion_database_path, experiment_name)
    df['orion_valid/HitRate1px'] = -df['objective']
    df = df[['id', 'status', 'orion_valid/HitRate1px']].rename(columns={'id': 'orion_id'})

    return df
