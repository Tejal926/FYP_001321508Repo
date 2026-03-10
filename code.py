import os
import time
import random
import pandas as pd
import discogs_client
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration
DISCOGS_USER_TOKEN = os.getenv("DISCOGS_USER_TOKEN","TwhKkMPvpWPwyZTsRCoPHHcabLZJJYddnKxmrXne")
if not DISCOGS_USER_TOKEN:
    raise ValueError("Set DISCOGS_USER_TOKEN in your environment (Discogs developer token).")

print("[INIT] Using Discogs token that is", "set" if DISCOGS_USER_TOKEN else "NOT set")

# Create Discogs client
print("[INIT] Creating Discogs client...")
d = discogs_client.Client("BookPlaylistApp/1.0", user_token=DISCOGS_USER_TOKEN)
print("[INIT] Discogs client created:", d)


# Books Dataset Upload
print("[INIT] Loading books Excel file...")
books_file = pd.read_excel('Backup_Books.xlsx')
print(f"[INIT] Loaded {len(books_file)} books from Backup_Books.xlsx")

# Sleep Section
RATE_LIMIT_BUFFER = 2      # keep 2 requests spare
MIN_SLEEP_SECONDS = 1.0    # always pause at least this long

def _respect_rate_limit(response):
    # Sleep if we're close to the Discogs per‑minute limit.
    headers = getattr(response, 'headers', {}) or {}

    try:
        limit = int(headers.get('X-Discogs-Ratelimit', 0))
        used = int(headers.get('X-Discogs-Ratelimit-Used', 0))
        remaining = int(headers.get('X-Discogs-Ratelimit-Remaining', 0))
    except (TypeError, ValueError):
        # If headers missing or not integers, just do a small fixed sleep
        time.sleep(MIN_SLEEP_SECONDS)
        return

    # If all headers are 0, they're not populated - just do light sleep
    if limit == 0 and used == 0 and remaining == 0:
        time.sleep(MIN_SLEEP_SECONDS)
        return

    print(f"[RATE] limit={limit}, used={used}, remaining={remaining}")

    # If we're running low on remaining calls in this 60‑second window,
    # sleep long enough for the window to reset.
    if remaining <= RATE_LIMIT_BUFFER:
        # conservative: wait for a full minute
        print("[RATE] Near limit; sleeping 60 seconds to reset window...")
        time.sleep(60)
    else:
        # light throttle so we don't burst
        time.sleep(MIN_SLEEP_SECONDS)

# Book commonalities (NLP)
def build_book_corpus(chosen_books: List[Dict]) -> Tuple[List[str], List[str]]:
    print("[CORPUS] Building corpus from chosen books...")
    titles = []
    corpus = []

    for idx, book in enumerate(chosen_books, start=1):
        title = str(book.get("Full Title", "Unknown title"))
        synopsis = str(book.get("Synopsis", "") or "")
        subjects = str(book.get("Subjects", "") or "")

        text = f"{synopsis} {subjects}".strip()
        if not text:
            text = title

        print(f"[CORPUS] Book #{idx}: '{title}' -> text length {len(text)}")
        titles.append(title)
        corpus.append(text)

    return titles, corpus

def book_commonalities(chosen_books: List[Dict]) -> None:
    # Read chosen_books['Synopsis'] and chosen_books['Subjects'],
    # use simple NLP (TF‑IDF + cosine similarity) to find common topics.

    print("\n[COMMON] Starting NLP analysis on chosen books...")
    titles, corpus = build_book_corpus(chosen_books)
    if len(corpus) < 2:
        print("[COMMON] Need at least two books to compare.")
        return

    vectorizer = TfidfVectorizer(stop_words="english",
                                 max_df=0.8,
                                 ngram_range=(1, 2),
                                 max_features=200)

    tfidf = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    print(f"[COMMON] TF‑IDF matrix shape: {tfidf.shape}")
    print(f"[COMMON] Number of features: {len(feature_names)}")

    # Global top keywords across all chosen books
    global_scores = tfidf.sum(axis=0).A1
    top_idx = global_scores.argsort()[::-1][:15]
    global_keywords = [(feature_names[i], round(global_scores[i], 3)) for i in top_idx]

    print("\nChosen books:")
    for i, t in enumerate(titles, 1):
        print(f"  {i}. {t}")

    print("\nTop shared topics/keywords:")
    for kw, score in global_keywords:
        print(f"  {kw} (score {score})")

    # Pairwise similarity
    print("\n[COMMON] Computing pairwise cosine similarity...")
    sim = cosine_similarity(tfidf, tfidf)

    print("\nPairwise similarity (cosine):")
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            print(f"  {titles[i]}  <->  {titles[j]} : {sim[i, j]:.3f}")

    # Build playlist from common topics
    print("\n[COMMON] Building playlist from common topics...\n")
    playlist = build_playlist_from_commonalities(global_keywords, total_tracks=15)

    print("\n" + "="*60)
    print("FINAL PLAYLIST")
    print("="*60)
    if playlist:
        for idx, track in enumerate(playlist, 1):
            print(f"{idx:2d}. {track['title']} – {track['artist']}  (theme: {track['reason']})")
            print(f"    Album: {track.get('release', 'N/A')}")
            print(f"    Verify: {track.get('discogs_url', 'N/A')}")
    else:
        print("[COMMON] Playlist is empty.")
    print("="*60)

BOOK_TOPIC_TO_DISCOGS_STYLE = {"science fiction": ["Electronic", "Score"],
                               "space opera": ["Space Rock", "Progressive Rock"],
                               "fantasy": ["Power Metal", "Symphonic Rock"],
                               "horror": ["Darkwave", "Goth Rock"],
                               "dystopian": ["Industrial", "EBM"],
                               "romance": ["Pop", "Ballad"],
                               "historical": ["Classical"],
                               "mystery": ["Jazz"]}

def pick_main_genres(global_keywords, max_genres=3):
    print("[GENRES] Selecting main genres from keywords...")
    genres = []
    for kw, _score in global_keywords:
        k = kw.lower()
        for topic in BOOK_TOPIC_TO_DISCOGS_STYLE.keys():
            if topic in k and topic not in genres:
                genres.append(topic)
                print(f"[GENRES] Matched keyword '{kw}' (score {_score}) -> topic '{topic}'")
                break
        if len(genres) >= max_genres:
            break
    # Fallback if nothing matched
    if not genres:
        genres = ["fantasy"]
        print("[GENRES] No genres matched keywords, using fallback 'fantasy'")
    else:
        print(f"[GENRES] Selected topics: {genres}")
    return genres


def search_discogs_tracks_for_genre(d, genre: str, limit: int = 5):
    print(f"[DISCOGS] Preparing to search for genre topic '{genre}' (limit {limit})...")
    style_options = BOOK_TOPIC_TO_DISCOGS_STYLE.get(genre, [])
    style = style_options[0] if style_options else None

    tracks = []

    if style:
        print(f"[DISCOGS] Using Discogs style '{style}' for topic '{genre}'")
        time.sleep(2.0)
        results = d.search(type="release", style=style, per_page=limit, page=1)
        _respect_rate_limit(results)
        print(f"[DISCOGS] Style search for '{style}' returned {len(results)} releases")
        tracks = _tracks_from_releases(results, genre, limit)
    else:
        print(f"[DISCOGS] No mapped style for topic '{genre}'")

    # Fallback: keyword in title if nothing found
    if not tracks:
        print(f"[DISCOGS] No tracks from style search, falling back to title search '{genre}'")
        time.sleep(2.0)
        results2 = d.search(type="release", title=genre, per_page=limit, page=1)
        _respect_rate_limit(results)
        print(f"[DISCOGS] Title search for '{genre}' returned {len(results2)} releases")
        tracks = _tracks_from_releases(results2, genre, limit)

    print(f"[DISCOGS] Returning {len(tracks)} tracks for topic '{genre}'\n")
    return tracks


def _tracks_from_releases(results, genre, max_tracks = 200):
    print(f"[TRACKS] Extracting up to {max_tracks} tracks for topic '{genre}'...")
    out = []
    count = 0
    for idx, release in enumerate(results, start=1):
        if len(out) >= max_tracks:
            print("[TRACKS] Reached requested max_tracks; stopping.")
            break

        try:
            # Fetch full release object to get tracklist
            release_id = release.id
            print(f"[TRACKS] Release #{idx}: Fetching full data for ID {release_id}...")
            full_release = d.release(release_id)
            _respect_rate_limit(full_release)
            time.sleep(2.0)

            title_release = full_release.title
            print(f"[TRACKS] '{title_release}'")

            tracklist = full_release.tracklist or []
            print(f"Tracklist length: {len(tracklist)}")
            if not tracklist:
                continue

            artist_name = full_release.artists[0].name.split(" (")[0] if full_release.artists else "Various"

            title = tracklist[0].title
            if not title:
                print("First track has no title, skipping.")
                continue

            # Verify the release exists with URL
            release_url = f"https://www.discogs.com/release/{release_id}"

            out.append({
                "title": title,
                "artist": artist_name,
                "reason": genre,
                "release": title_release,
                "discogs_url": release_url,
                "release_id": release_id
            })
            count += 1
            print(f"Added track: '{title}' – {artist_name} (total so far: {count})")
            print(f"  Release: {title_release}")
            print(f"  Verify at: {release_url}")

        except Exception as e:
            print(f"[TRACKS] Error fetching release #{idx}: {e}")
            continue

    print(f"[TRACKS] Finished extracting tracks for '{genre}'. Collected {len(out)}.\n")
    return out


def build_playlist_from_commonalities(global_keywords, total_tracks: int = 15):
    print(f"[PLAYLIST] Building playlist for total_tracks={total_tracks}...")
    genres = pick_main_genres(global_keywords, max_genres=3)

    print(f"[PLAYLIST] Genres to use: {genres}")
    if not genres:
        genres = ["fantasy"]
        print("[PLAYLIST] Genres list empty, forcing ['fantasy']")

    per_theme = max(1, total_tracks // len(genres))
    print(f"[PLAYLIST] Will request {per_theme} tracks per genre")

    playlist = []
    for i, genre in enumerate(genres):
        print(f"[PLAYLIST] === Genre {i + 1}/{len(genres)}: '{genre}' ===")
        tracks = search_discogs_tracks_for_genre(d, genre, limit=per_theme)
        print(f"[PLAYLIST] Got {len(tracks)} tracks for '{genre}'")
        playlist.extend(tracks)
        print(f"[PLAYLIST] Playlist length is now {len(playlist)}")
        if i < len(genres) - 1:
            print("[PLAYLIST] Sleeping 2 seconds before next genre...")
            time.sleep(2.0)

    if not playlist:
        print("[PLAYLIST] No tracks found from Discogs - playlist empty.")
    else:
        print(f"[PLAYLIST] Built playlist with {len(playlist)} total tracks.")

    return playlist[:total_tracks]


def main():
    try:

        print("\n[MAIN] Available books:")
        for idx, row in books_file.iterrows():
            print(f"{idx + 1}. {row['Full Title']}")

        books_num = int(input("\n[MAIN] How many books will you select? "))
        chosen_rows = []
        for i in range(books_num):
            while True:
                try:
                    choice = int(input(f"Select book #{i + 1} (1-{len(books_file)}): "))
                    if 1 <= choice <= len(books_file):
                        chosen_rows.append(books_file.iloc[choice - 1].to_dict())
                        break
                    else:
                        print("[MAIN] Out of range, try again.")
                except ValueError:
                    print("[MAIN] Please enter a number.")

        print("[MAIN] All books chosen. Running commonalities + playlist generation...")
        book_commonalities(chosen_rows)

        #test_release = d.release(random.randint(1000,100000))
        #print("Test track:", test_release.tracklist[0].title, "–", test_release.artists[0].name)

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()



