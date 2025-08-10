import pandas as pd
from backend.main import compute_simple_exposure

def test_compute_simple_exposure_basic():
    df = pd.DataFrame([
        {"ticker": "AAA", "quantity": 10, "price": 2.5, "beta": 1.2},
        {"ticker": "BBB", "quantity": 5, "price": 10.0, "beta": 0.5},
    ])
    res = compute_simple_exposure(df)
    # exposure = 10*2.5*1.2 + 5*10*0.5 = 30 + 25 = 55
    assert abs(res["total_exposure"] - 55.0) < 1e-6
