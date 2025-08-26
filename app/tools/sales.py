import pandas as pd

def get_historical_sales(client: str):
    df = pd.read_csv("data/sales.csv")
    client_sales = df[df["client"].str.lower() == client.lower()]
    if client_sales.empty:
        return f"No sales found for {client}"
    return client_sales.to_dict(orient="records")