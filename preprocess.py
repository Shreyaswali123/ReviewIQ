import ftfy
import emoji
import re
from langdetect import detect, LangDetectException
from datasketch import MinHash, MinHashLSH
from collections import defaultdict

def clean_text(text: str) -> str:
    text = ftfy.fix_text(str(text))
    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_language(text: str) -> str:
    if re.search(r'\b(hai|kya|tha|yeh|aur|bhai|karo|bhot|mast|ekdum)\b', text, re.IGNORECASE):
        return "hinglish"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def create_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=128)
    for word in text.lower().split():
        m.update(word.encode('utf8'))
    return m

def detect_bots(reviews: list) -> dict:
    lsh = MinHashLSH(threshold=0.92, num_perm=128)
    minhashes = {}

    for i, r in enumerate(reviews):
        review_id = str(r.get('review_id', r.get('id', f"rev_{i}")))
        text = str(r.get('review_text', r.get('text', '')))
        if len(text.split()) < 4:
            continue
        m = create_minhash(text)
        minhashes[review_id] = m

    inserted = set()
    raw_clusters = {}
    for review_id, m in minhashes.items():
        result = lsh.query(m)
        if result:
            raw_clusters[review_id] = result
        if review_id not in inserted:
            try:
                lsh.insert(review_id, m)
                inserted.add(review_id)
            except ValueError:
                pass

    adjacency = defaultdict(set)
    for rid, matches in raw_clusters.items():
        for match in matches:
            adjacency[rid].add(match)
            adjacency[match].add(rid)

    visited = set()
    components = []

    def bfs(start):
        cluster = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.add(node)
            queue.extend(adjacency[node] - visited)
        return cluster

    for node in adjacency:
        if node not in visited:
            comp = bfs(node)
            if len(comp) >= 3:
                components.append(comp)

    bot_clusters = {}
    spam_keywords = r'\b(amazing product|best item|excellent functionality|very nice product|fast shipping|top seller|highly recommended|will buy again)\b'

    for comp in components:
        comp_list = list(comp)
        
        sample_rid = comp_list[0]
        sample_text = str(next((r.get('review_text', r.get('text', '')) for r in reviews if str(r.get('review_id', r.get('id', ''))) == sample_rid), ''))
        
        # Only flag if it has spam keywords OR multiple uppercase words (like "AMAZING PRODUCT")
        is_spam_like = bool(re.search(spam_keywords, sample_text, re.IGNORECASE)) or len(re.findall(r'\b[A-Z]{3,}\b', sample_text)) >= 2
        
        if is_spam_like:
            for rid in comp_list:
                bot_clusters[rid] = [x for x in comp_list if x != rid]

    return bot_clusters