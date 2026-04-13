import math
import requests
from datetime import datetime

def scrape_twitter_user(username):
    """
    Fetches Twitter user data using the public fxtwitter API, completely 
    bypassing the need for authentication, Selenium, or Paid API keys.
    """
    try:
        response = requests.get(f"https://api.fxtwitter.com/{username}", timeout=10)
        
        if response.status_code != 200:
            raise ValueError(f"Account @{username} not found or suspended.")

        r_json = response.json()
        if 'user' not in r_json:
            raise ValueError(f"Could not parse data for @{username}.")
            
        user = r_json['user']

        # Extract primary metrics
        f = user.get("followers", 0)
        fg = user.get("following", 0)
        t = user.get("tweets", 0)
        fav = user.get("likes", 200) # Use actual likes!
        
        # Determine verification and defaults
        verify_obj = user.get("verification", {})
        v = 1 if verify_obj.get("verified", False) else 0
        
        avatar_url = user.get("avatar_url", "")
        dpi = 1 if avatar_url and "default_profile" in avatar_url else 0
        
        # Calculate Account Age
        joined_str = user.get("joined")
        # Format is 'Fri Oct 03 04:16:17 +0000 2008'
        try:
            created_at = datetime.strptime(joined_str, "%a %b %d %H:%M:%S %z %Y")
            account_age_days = (datetime.now(created_at.tzinfo) - created_at).days
            account_age_days = max(account_age_days, 1) # Prevent div by 0
            joined_date = created_at.strftime('%Y-%m-%d')
        except:
            account_age_days = 365
            joined_date = "Unknown"

        has_bio = bool(user.get("description", ""))

        # Build exactly as expected by ML and UI
        data = {
            "profile": {
                "username": username,
                "display_name": user.get("name", username),
                "avatar_url": avatar_url if avatar_url and "default_profile" not in avatar_url else "",
                "followers": f,
                "following": fg,
                "tweets": t,
                "bio": user.get("description", ""),
                "location": user.get("location", ""),
                "joined": joined_date,
                "verified": bool(v),
                "has_bio": has_bio,
                "pinned_tweet": False
            },
            "features": {
                "followers_count": f,
                "friends_count": fg,
                "statuses_count": t,
                "favourites_count": fav,
                "verified": v,
                "default_profile_image": dpi,
                "account_age_days": account_age_days,
                
                "followers_friends_ratio": f / (fg + 1),
                "statuses_per_day": t / account_age_days,
                "favourites_per_status": fav / (t + 1),
                
                "log_followers": math.log(f + 1),
                "log_friends": math.log(fg + 1),
                "log_statuses": math.log(t + 1),
            }
        }
        return data

    except Exception as e:
        raise ValueError(f"Free API Error: {str(e)}")
