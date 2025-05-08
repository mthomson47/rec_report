import re
import pandas as pd
import streamlit as st

MONTH_TO_CODE = {
    "JAN": "F", "FEB": "G", "MAR": "H", "APR": "J",
    "MAY": "K", "JUN": "M", "JUL": "N", "AUG": "Q",
    "SEP": "U", "OCT": "V", "NOV": "X", "DEC": "Z",
}
REGION_TO_CODE = {"TTF": 1, "HH": 2, "WTI": 4, "BRENT": 5}
CODE_TO_MONTH = {v: k for k, v in MONTH_TO_CODE.items()}
CODE_TO_REGION = {v: k for k, v in REGION_TO_CODE.items()}

def contract_to_ticker(contract: str) -> str:
    m = re.match(r"^(\w+)\s+([A-Za-z]+)-(\d{2})$", contract.strip())
    if not m:
        raise ValueError(f"Invalid contract format: {contract!r}")
    region, mon, yy = m.group(1).upper(), m.group(2).upper(), m.group(3)
    mon_code = MONTH_TO_CODE[mon[:3]]
    reg_code = REGION_TO_CODE[region]
    return f"{mon_code}_{yy}_{reg_code}"

def ticker_to_contract(ticker: str) -> str:
    parts = ticker.strip().split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid ticker format: {ticker!r}")
    mon_code, yy, reg_code_str = parts
    month = CODE_TO_MONTH[mon_code]
    region = CODE_TO_REGION[int(reg_code_str)]
    return f"{region} {month.capitalize()}-{yy}"

def transform_recs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Strike'] = (
        df['STRIKE'].astype(str)
                   .str.replace(',', '.')
                   .astype(float)
    )

    contracts, positions, deltas = [], [], []
    fair_values, best_bids, best_offers = [], [], []

    for q in df['QUOTE']:
        tok = q.split()
        contracts.append(f"{tok[0]} {tok[1]}")

        m = re.search(r'^\S+\s+\S+\s+([^(]+?)(?:\s*\(|\s+strike)', q)
        positions.append(m.group(1).strip() if m else None)

        m = re.search(r'delta X ([\d,]+)', q)
        deltas.append(float(m.group(1).replace(',', '.')) if m else None)

        m = re.search(r'fair value ([\d,]+)', q)
        fair_values.append(float(m.group(1).replace(',', '.')) if m else None)

        mb = re.search(r'best bid ([\d,]+)', q)
        mo = re.search(r'best offer ([\d,]+)', q)
        best_bids.append(float(mb.group(1).replace(',', '.')) if mb else None)
        best_offers.append(float(mo.group(1).replace(',', '.')) if mo else None)

    result = pd.DataFrame({
        'ID':        df['ID'],
        'Contract':  contracts,
        'Position':  positions,
        'Strike':    df['Strike'],
        'Underlying':     deltas,
        'Fair Value':fair_values,
        'Best Bid':  best_bids,
        'Best Offer':best_offers,
        'Put/Call':  df['PUT_CALL'],
        'Long/Short':df['LONG_SHORT']
    })

    return result.reset_index(drop=True)

