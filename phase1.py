import os
import random
import re
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. CONFIGURATION & TEMPLATES  (module-level constants only)
# ---------------------------------------------------------

OUTPUT_DIR = "backend/data/synthetic/"

B1_START = datetime(2026, 1, 1)
B1_END   = datetime(2026, 2, 14)
B2_START = datetime(2026, 2, 15)
B2_END   = datetime(2026, 3, 31)

SW_POS = [
    "Absolutely love this smartwatch! The display is crisp and bright.",
    "Good value for money. Tracks my steps accurately.",
    "Design is premium. Very happy with the GT Pro 5.",
    "It's a solid watch, does everything I need for my workouts.",
    "Fantastic purchase. Highly recommend it to anyone.",
    "The heart rate monitor seems very accurate compared to my chest strap.",
    "Sleek design and comfortable strap for all-day wear.",
    "Notifications sync perfectly with my phone. Zero lag.",
    "Water resistance holds up great during my swimming sessions.",
    "Great alternative to the much more expensive competitors."
]
SW_BATT_SUBTLE = [
    "Good watch overall, though the battery drains a bit fast if GPS is left on.",
    "Nice display and features, I have to charge it more than expected but it's okay.",
    "Love the fitness tracking, just wish the battery lasted a tiny bit longer.",
    "Wish I didn't have to pack the charger for weekend trips.",
    "Screen is great, but it definitely eats into the power reserves quickly.",
    "A bit annoying to charge every single night, but otherwise it's a fine device."
]
SW_BATT_DIRECT = [
    "Battery life is terrible since the update!",
    "Since the update, it dies in 4 hours. Unusable.",
    "Ruined since the update. Battery drains instantly now.",
    "Do not buy. Since the update battery is 0% in half a day.",
    "The new firmware update completely destroyed the battery capacity.",
    "I used to get 3 days, now since the update I barely get 8 hours."
]
PB_POS = [
    "Tastes amazing and keeps me full during my shifts.",
    "Best protein bar I've had. Macros are great.",
    "Delicious flavor. Will definitely buy again.",
    "Good texture, not chalky at all like other brands.",
    "Perfect post-workout snack.",
    "Doesn't leave that weird artificial aftertaste. Highly recommended.",
    "Great for a quick breakfast on the go when I'm running late.",
    "The peanut butter crunch flavor is absolutely out of this world.",
    "Finally, a protein bar that doesn't feel like chewing on cardboard.",
    "Super filling! Fits perfectly into my daily calorie goals."
]
PB_PACKAGING = [
    "Box packaging was crushed and bars were slightly melted.",
    "Terrible packaging, half the bars in the box were smashed.",
    "Product is okay but the packaging was completely ruined.",
    "Hard to open the wrapper, and bad packaging overall.",
    "The delivery box was fine but the product box inside was torn to shreds.",
    "Seals on a few of the individual bars were open. Unacceptable packaging."
]
SH_POS = [
    "Very comfortable for daily runs and gym sessions.",
    "Fit perfectly, true to size.",
    "Great cushioning and surprisingly lightweight.",
    "Best running shoes I've bought this year.",
    "Awesome grip on wet surfaces.",
    "The arch support is exactly what I needed for my long distance runs.",
    "Breathable material keeps my feet cool even in summer.",
    "Stylish enough to wear casually and functional enough for the track.",
    "No break-in period required, they felt great right out of the box.",
    "Excellent bounce and energy return on the pavement."
]
SH_COMPLAINT = [
    "Sole wore out quickly after just 50 miles.",
    "A bit too tight on the toes, definitely size up.",
    "Gave me bad blisters on the heel.",
    "Not as durable as I hoped for the price.",
    "The laces keep untying themselves no matter what knot I use.",
    "Colors look much more faded in person than in the product photos."
]
SARCASM = {
    'GT Pro 5': [
        "Oh brilliant, another smartwatch that becomes a paperweight.",
        "Wow, I love charging my watch 3 times a day!",
        "Such an advanced piece of tech, it helps me practice plugging things in constantly."
    ],
    'NutriMax': [
        "Yum, I love eating crushed dust instead of a solid bar.",
        "Ah yes, the 'broken box' flavor is my absolute favorite.",
        "Nothing starts the day better than a protein bar that looks like someone sat on it."
    ],
    'AeroGlide': [
        "Perfect shoes if you enjoy getting blisters instantly.",
        "Amazing durability, if you only plan to run exactly once.",
        "I didn't know I was paying for medieval torture devices for my feet."
    ]
}
BOT_BASES = [
    "AMAZING PRODUCT VERY GOOD QUALITY WILL BUY AGAIN",
    "best item ever received fast shipping love it",
    "Excellent functionality perfect condition top seller",
    "very nice product working fine highly recommended",
    "A+++ SELLER GREAT COMMUNICATION FAST DELIVERY",
    "wow just wow this is the best purchase of my life"
]
DELAYS = [
    " Arrived late.", " Took 12 days to arrive.", " Delivery was delayed.",
    " Shipping took forever.", " Courier lost the package for a week before it showed up.",
    " Prime shipping? More like prehistoric shipping times."
]


# ---------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------

def _rdate(start, end, rng):
    return start + timedelta(days=rng.randint(0, (end - start).days))

def _hinglish(text):
    words = {"good": "ekdum mast", "bad": "kharab", "worst": "bekar", "love": "pasand", "great": "bhot badhiya"}
    changed = False
    for k, v in words.items():
        if re.search(r'\b' + k + r'\b', text, re.IGNORECASE):
            text = re.sub(r'\b' + k + r'\b', v, text, re.IGNORECASE); changed = True
    return text if changed else text + " Bhot badhiya."

def _typos(text):
    typos = {"comfortable": "comfertable", "quality": "qualty", "terrible": "turrible",
             "awesome": "awsome", "display": "displey", "flavor": "flavourr"}
    changed = False
    for k, v in typos.items():
        if re.search(r'\b' + k + r'\b', text, re.IGNORECASE):
            text = re.sub(r'\b' + k + r'\b', v, text, re.IGNORECASE); changed = True
    return text if changed else text + " Hopfuly."

def _emojis(text, rng):
    return text + " " + rng.choice(["🔥", "💀", "📦💔", "👟✨", "😡", "💯", "👎", "🙄", "🤩", "🔋📉"])


# ---------------------------------------------------------
# 3. CORE GENERATOR  (pure function, no file I/O)
# ---------------------------------------------------------

def generate_demo_reviews(seed: int = 42) -> list[dict]:
    """
    Generates the full 500-review demo dataset in memory.
    Safe to import — no file writes, no global state mutated.
    """
    rng     = random.Random(seed)
    counter = [1]

    def make(prod, cat, text, rating, date, bot=False, sarc=False):
        r = {'review_id': f"REV{counter[0]:04d}", 'product_name': prod, 'category': cat,
             'review_text': text, 'star_rating': rating,
             'review_date': date.strftime('%Y-%m-%d'),
             'reviewer_id': f"USR{rng.randint(1000,9999)}",
             'is_bot': bot, 'is_sarcastic': sarc}
        counter[0] += 1
        return r

    sw, pb, sh = [], [], []

    # Smartwatch — Batch 1: 6% battery; Batch 2: 38% battery
    for _ in range(6):  sw.append(make("GT Pro 5","Electronics",rng.choice(SW_BATT_SUBTLE),rng.randint(3,4),_rdate(B1_START,B1_END,rng)))
    for _ in range(94): sw.append(make("GT Pro 5","Electronics",rng.choice(SW_POS),         rng.randint(4,5),_rdate(B1_START,B1_END,rng)))
    for _ in range(38): sw.append(make("GT Pro 5","Electronics",rng.choice(SW_BATT_DIRECT), rng.randint(1,2),_rdate(B2_START,B2_END,rng)))
    for _ in range(62): sw.append(make("GT Pro 5","Electronics",rng.choice(SW_POS),         rng.randint(4,5),_rdate(B2_START,B2_END,rng)))

    # Protein Bar — Batch 1: 5% packaging; Batch 2: 35% packaging
    for _ in range(4):  pb.append(make("NutriMax","Food",rng.choice(PB_PACKAGING),rng.randint(1,3),_rdate(B1_START,B1_END,rng)))
    for _ in range(76): pb.append(make("NutriMax","Food",rng.choice(PB_POS),      rng.randint(4,5),_rdate(B1_START,B1_END,rng)))
    for _ in range(28): pb.append(make("NutriMax","Food",rng.choice(PB_PACKAGING),rng.randint(1,3),_rdate(B2_START,B2_END,rng)))
    for _ in range(52): pb.append(make("NutriMax","Food",rng.choice(PB_POS),      rng.randint(4,5),_rdate(B2_START,B2_END,rng)))

    # Shoes — spread across both batches
    for _ in range(100): sh.append(make("AeroGlide","Apparel",rng.choice(SH_POS),      rng.randint(4,5),_rdate(B1_START,B2_END,rng)))
    for _ in range(40):  sh.append(make("AeroGlide","Apparel",rng.choice(SH_COMPLAINT),rng.randint(1,3),_rdate(B1_START,B2_END,rng)))

    all_data = sw + pb + sh  # 500 reviews

    # Cross-product delays
    for idx in rng.sample(range(len(all_data)), 55):
        all_data[idx]['review_text'] += rng.choice(DELAYS)

    # Sarcasm
    for ds, key in zip([sw, pb, sh], ['GT Pro 5', 'NutriMax', 'AeroGlide']):
        cands = [i for i, d in enumerate(ds) if d['star_rating'] == 5]
        for idx in rng.sample(cands, 6):
            ds[idx]['review_text'] = rng.choice(SARCASM[key])
            ds[idx]['star_rating'] = rng.randint(1, 2)
            ds[idx]['is_sarcastic'] = True

    # Bot clusters
    bot_cands = [i for i, d in enumerate(all_data) if d['star_rating']==5 and not d['is_sarcastic']]
    bot_idxs  = rng.sample(bot_cands, 24)
    for cid in range(6):
        base = BOT_BASES[cid]
        for j, idx in enumerate(bot_idxs[cid*4:(cid+1)*4]):
            all_data[idx]['review_text'] = base + ("!" * j)
            all_data[idx]['star_rating'] = 5
            all_data[idx]['is_bot'] = True

    # Linguistic noise every 8th review
    noise_funcs = [_hinglish, _emojis, _typos]
    ni = 0
    for i in range(7, len(all_data), 8):
        item = all_data[i]
        if not item['is_bot'] and not item['is_sarcastic']:
            if not any(k in item['review_text'].lower() for k in ['battery','packaging','update']):
                fn = noise_funcs[ni % 3]
                item['review_text'] = fn(item['review_text'], rng) if fn == _emojis else fn(item['review_text'])
                ni += 1

    return all_data


# ---------------------------------------------------------
# 4. SCRIPT MODE
# ---------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_reviews = generate_demo_reviews()
    sw_r = [r for r in all_reviews if r['product_name'] == 'GT Pro 5']
    pb_r = [r for r in all_reviews if r['product_name'] == 'NutriMax']
    sh_r = [r for r in all_reviews if r['product_name'] == 'AeroGlide']
    pd.DataFrame(sw_r).to_csv(os.path.join(OUTPUT_DIR, "smartwatch_reviews.csv"), index=False)
    pd.DataFrame(pb_r).to_csv(os.path.join(OUTPUT_DIR, "protein_bar_reviews.csv"), index=False)
    pd.DataFrame(sh_r).to_csv(os.path.join(OUTPUT_DIR, "running_shoes_reviews.csv"), index=False)
    print(f"Generated {len(all_reviews)} reviews → {OUTPUT_DIR}")