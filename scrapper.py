from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time, math

def scrape_twitter_user(username):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(f"https://twitter.com/{username}")
        time.sleep(4)

        def get_text(selector):
            try:
                return driver.find_element(By.CSS_SELECTOR, selector).text
            except:
                return ""

        followers = get_text('[data-testid="UserProfileHeader_Items"] a[href*="followers"] span span')
        following = get_text('[data-testid="UserProfileHeader_Items"] a[href*="following"] span span')
        tweets    = get_text('[data-testid="UserProfileHeader_Items"] a[href*="tweets"] span span')
        bio       = get_text('[data-testid="UserDescription"]')
        name      = get_text('[data-testid="UserName"] span')
        location  = get_text('[data-testid="UserLocation"]')
        created_at = get_text('time')

        def to_num(s):
            s = s.replace(',','').strip()
            if 'K' in s: return int(float(s.replace('K',''))*1000)
            if 'M' in s: return int(float(s.replace('M',''))*1000000)
            try: return int(s)
            except: return 0

        f  = to_num(followers)
        fg = to_num(following)
        t  = to_num(tweets)
        bio_words = len(bio.split()) if bio else 0
        account_age_days = 365

        data = {
    # ✅ PROFILE (for UI)
    "profile": {
        "username": username,
        "display_name": name,
        "followers": f,
        "following": fg,
        "tweets": t,
        "bio": bio,
        "location": location,
    },

    # ✅ FEATURES (for ML model)
    "features": {
    "followers_count": f,
    "friends_count": fg,
    "statuses_count": t,
    "favourites_count": 200,  # placeholder
    "verified": 0,
    "default_profile_image": 0,
    "account_age_days": account_age_days,

    "followers_friends_ratio": f / (fg + 1),
    "statuses_per_day": t / 365,
    "favourites_per_status": 200 / (t + 1),

    "log_followers": math.log(f + 1),
    "log_friends": math.log(fg + 1),
    "log_statuses": math.log(t + 1),

    }
}
        return data
    finally:
        driver.quit()
