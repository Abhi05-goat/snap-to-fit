import math
from confidence import get_confidence

def placement_fit(item: dict, space: dict, margin_cm: float = 15.0) -> dict:
    if not item.get("ok") or not space.get("ok"):
        return {"ok": False, "error": "Invalid item or space"}

    L, W, H = item.get("L_cm", 0), item.get("W_cm", 0), item.get("H_cm", 0)
    item_sigma = item.get("sigma_pct", 0.0)
    
    S_W, S_D = space.get("width_cm", 0), space.get("depth_cm", 0)
    space_sigma = space.get("sigma_pct", 0.0)
    
    # Try 0 degrees: L along S_W, W along S_D
    margin_0_W = S_W - L
    margin_0_D = S_D - W
    margin_0 = min(margin_0_W, margin_0_D)
    
    # Try 90 degrees: W along S_W, L along S_D
    margin_90_W = S_W - W
    margin_90_D = S_D - L
    margin_90 = min(margin_90_W, margin_90_D)
    
    if margin_0 >= margin_90:
        best_margin = margin_0
        best_orientation = "0 degrees (Length along width, Width along depth)"
        if margin_0_W < margin_0_D:
            limiting_item_dim, limiting_space_dim = L, S_W
        else:
            limiting_item_dim, limiting_space_dim = W, S_D
    else:
        best_margin = margin_90
        best_orientation = "90 degrees (Width along width, Length along depth)"
        if margin_90_W < margin_90_D:
            limiting_item_dim, limiting_space_dim = W, S_W
        else:
            limiting_item_dim, limiting_space_dim = L, S_D
            
    if best_margin >= margin_cm:
        verdict = "fits"
    elif best_margin >= 0:
        verdict = "tight"
    else:
        verdict = "doesn't fit"
        
    prob_fit, tier = get_confidence(best_margin, limiting_item_dim, item_sigma, limiting_space_dim, space_sigma)
    
    return {
        "ok": True,
        "verdict": verdict,
        "margin_cm": best_margin,
        "detail": best_orientation,
        "prob_fit": prob_fit,
        "tier": tier
    }

def passage_fit(item: dict, space: dict, margin_cm: float = 15.0) -> dict:
    if not item.get("ok") or not space.get("ok"):
        return {"ok": False, "error": "Invalid item or space"}
        
    L, W, H = item.get("L_cm", 0), item.get("W_cm", 0), item.get("H_cm", 0)
    item_sigma = item.get("sigma_pct", 0.0)
    
    S_W, S_H = space.get("width_cm", 0), space.get("height_cm", 0)
    space_sigma = space.get("sigma_pct", 0.0)
    
    pairs = [
        ("W, H", W, H),
        ("L, H", L, H),
        ("L, W", L, W)
    ]
    
    best_margin = -float('inf')
    best_detail = ""
    limiting_item_dim = 0
    limiting_space_dim = 0
    
    for name, w, h in pairs:
        for angle_deg in range(91):
            theta = math.radians(angle_deg)
            req_w = w * math.cos(theta) + h * math.sin(theta)
            req_h = w * math.sin(theta) + h * math.cos(theta)
            
            margin_w = S_W - req_w
            margin_h = S_H - req_h
            
            current_margin = min(margin_w, margin_h)
            if current_margin > best_margin:
                best_margin = current_margin
                best_detail = f"Cross-section {name} tilted at {angle_deg} degrees"
                # Heuristic for limiting dim:
                if margin_w < margin_h:
                    limiting_item_dim = w if math.cos(theta) > math.sin(theta) else h
                    limiting_space_dim = S_W
                else:
                    limiting_item_dim = h if math.cos(theta) > math.sin(theta) else w
                    limiting_space_dim = S_H

    if best_margin >= margin_cm:
        verdict = "fits"
    elif best_margin >= 0:
        verdict = "tight"
    else:
        verdict = "doesn't fit"
        
    prob_fit, tier = get_confidence(best_margin, limiting_item_dim, item_sigma, limiting_space_dim, space_sigma)
        
    return {
        "ok": True,
        "verdict": verdict,
        "margin_cm": best_margin,
        "detail": best_detail,
        "prob_fit": prob_fit,
        "tier": tier
    }
