import boto3
import pandas as pd
from io import StringIO
from app.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def GetMonteCarloForecast(commodity: str, days: int = 5) -> dict:
    """
    Get price range simulation based in Montecarlo Method
    """
    key = f"{commodity.capitalize()}_sim.csv"
    # key = f"Copper_sim.csv"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    if commodity.capitalize() == 'Copper':
        df['mean'] = df['mean'] / 0.453592  # Convert to kg
        df['upper'] = df['upper'] / 0.453592  # Convert to kg
        df['lower'] = df['lower'] / 0.453592  # Convert to kg
    elif commodity.capitalize() == 'Zinc':
        df['mean'] = df['mean'] / 1000  # Convert to kg
        df['upper'] = df['upper'] / 1000  # Convert to kg
        df['lower'] = df['lower'] / 1000  # Convert to kg
    
    row = df[df["days"] == days]
    if row.empty:
        return f"No forecast found for {commodity} at horizon {days} days"
    

    r = row.iloc[0]
    return {
        "commodity": commodity,
        "days": int(r["days"]),
        "mean": float(r["mean"]),
        "upper": float(r["upper"]),
        "lower": float(r["lower"]),
    }

def getForecastPrice(commodity: str, ):
    """
    Get price forecast for a specific commodity.
    """
    key = f"{commodity.capitalize()}_forecast_price.csv"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    if commodity.capitalize() == 'Copper':
        df['last_close'] = df['last_close'] / 0.453592  # Convert to kg
        df['pred_upper_price'] = df['pred_upper_price'] / 0.453592  # Convert to kg
        df['pred_lower_price'] = df['pred_lower_price'] / 0.453592  # Convert to kg
    elif commodity.capitalize() == 'Zinc':
        df['last_close'] = df['last_close'] / 1000  # Convert to kg
        df['pred_upper_price'] = df['pred_upper_price'] / 1000  # Convert to kg
        df['pred_lower_price'] = df['pred_lower_price'] / 1000  # Convert to kg

    r = df.iloc[0]
    return {
        "commodity": commodity,
        "date": str(r["date"]),
        "close": float(r["last_close"]),
        "upper_return": float(r["pred_upper_ret"]),
        "lower_return": float(r["pred_lower_ret"]),
        "upper_price": float(r["pred_upper_price"]),
        "lower_price": float(r["pred_lower_price"]),
        "coverage_rate": float(r["coverage_rate"])

    }

def getForecastVolatility(commodity: str):
    """
    Get volatility forecast for a specific commodity.
    """
    key = f"{commodity.capitalize()}_forecast_volatility.csv"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    if len(df) < 10:
        raise ValueError(f"Forecast volatility for {commodity} has less than 10 days of data.")

    # Extraemos valores relevantes
    vol_1d = float(df.iloc[0]["forecast_volatility"])
    vol_5d = float(df.iloc[4]["forecast_volatility"])
    vol_10d = float(df.iloc[9]["forecast_volatility"])
    vol_last = float(df.iloc[-1]["forecast_volatility"])

    change = (vol_last - vol_1d) / vol_1d

    # Criterio simple de trending
    if change > 0.2:  
        trending = "high increase"
    elif change > 0.05:
        trending = "increase"
    elif change > -0.05:
        trending = "neutral"
    elif change > -0.2:
        trending = "decrease"
    else:
        trending = "high decrease"

    return {
        "commodity": commodity.capitalize(),
        "volatility_predict_1d": vol_1d,
        "volatility_predict_5d": vol_5d,
        "volatility_predict_10d": vol_10d,
        "volatility_last": vol_last,
        "trending": trending
    }