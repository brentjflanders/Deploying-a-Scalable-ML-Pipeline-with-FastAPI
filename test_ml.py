import pytest
import pandas as pd
from ml.model import train_model
from sklearn.ensemble import RandomForestClassifier
from ml.model import compute_model_metrics


# TODO: implement the first test. Change the function name and input as needed
def test_model_type():
    """
    Verify that we are using the expected model type (RandomForestClassifier)
    """
    unit_test_x = [[1,2,3],[4,5,6]]
    unit_test_y = ['col_A','col_B']

    model = train_model(unit_test_x, unit_test_y)

    assert isinstance(model, RandomForestClassifier)
    

def test_col_names():
    """
    Verify that all expected columns are present in the data
    """
    df = pd.read_csv('./data/census.csv')

    expected_features = {
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'salary'
    }
    
    assert set(df.columns) == expected_features


def test_model_metrics():
    """
    Verify that precision, recall, and F-Beta metrics are of the expected datatype (float)
    and fall within expected ranges (0-1)
    """
    y = [1,1,0,1]
    preds = [1,0,1,0]

    precision,recall,fbeta = compute_model_metrics(y,preds)

    assert isinstance(precision,float)
    assert isinstance(recall,float)
    assert isinstance(fbeta,float)

    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(1/3)
    assert fbeta == pytest.approx(0.4)
