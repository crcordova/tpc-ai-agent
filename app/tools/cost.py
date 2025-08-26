def calculate_cost(components: list, forecasts: dict):
    """
    Calcula costo ponderado según porcentajes de componentes.
    components: [{"commodity": "copper", "percentage": 50}, ...]
    forecasts: {"copper": {"mean": ..., "upper": ..., "lower": ...}, ...}
    """
    total_cost = 0
    breakdown = []

    for comp in components:
        c = comp["commodity"].lower()
        pct = comp["percentage"] / 100
        if c not in forecasts:
            breakdown.append(f"{c}: no forecast found")
            continue
        
        mean_price = forecasts[c]["mean"]
        contrib = mean_price * pct
        total_cost += contrib
        breakdown.append(f"{comp['commodity']}({comp['percentage']}%): {mean_price:.2f} USD/t")

    return {
        "estimated_cost": round(total_cost, 2),
        "details": breakdown
    }
