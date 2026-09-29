import math

SIGMA_FLOOR = 0.08 # retuned in Phase 10

def get_confidence(margin_cm: float, item_dim: float, item_sigma_pct: float, space_dim: float, space_sigma_pct: float):
    sigma_item = max(item_sigma_pct, SIGMA_FLOOR)
    sigma_space = max(space_sigma_pct, SIGMA_FLOOR)
    
    margin_std = math.sqrt((sigma_item * item_dim)**2 + (sigma_space * space_dim)**2)
    if margin_std == 0:
        prob_fit = 1.0 if margin_cm >= 0 else 0.0
    else:
        # prob_fit = Φ(margin / margin_std)
        # using math.erf: Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
        prob_fit = 0.5 * (1 + math.erf(margin_cm / (margin_std * math.sqrt(2))))
    
    certainty = max(prob_fit, 1.0 - prob_fit)
    if certainty >= 0.90:
        tier = "High"
    elif certainty >= 0.70:
        tier = "Medium"
    else:
        tier = "Low"
        
    return prob_fit, tier
