import os
import time
import json
import math
import pandas as pd
import discogs_client
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration
DISCOGS_USER_TOKEN = os.getenv("DISCOGS_USER_TOKEN", "TwhKkMPvpWPwyZTsRCoPHHcabLZJJYddnKxmrXne")
RATE_LIMIT_BUFFER = 2
MIN_SLEEP_SECONDS = 1.0
STYLES_CACHE_FILE = "discogs_styles_cache.json"

# Initialize
print("Loading books and Discogs client...")
books_file = pd.read_excel('Backup_Books.xlsx')
d = discogs_client.Client("BookPlaylistApp/1.0", user_token=DISCOGS_USER_TOKEN)
print(f"Loaded {len(books_file)} books\n")

def _respect_rate_limit(response):
    # Handle Discogs API rate limiting.
    headers = getattr(response, 'headers', {}) or {}
    try:
        remaining = int(headers.get('X-Discogs-Ratelimit-Remaining', 0))
        if remaining <= RATE_LIMIT_BUFFER:
            print("Approaching rate limit, waiting 60 seconds...")
            time.sleep(60)
            return
    except (TypeError, ValueError):
        pass
    time.sleep(MIN_SLEEP_SECONDS)

# Genre descriptions for semantic matching
GENRE_DESCRIPTIONS = {"science fiction": "space alien robot future technology planet android "
                                         "spaceship cosmos extraterrestrial sci-fi spacecraft galaxy interstellar",
                      "cyberpunk": "cyber tech hacker dystopia corporate virtual digital network matrix "
                                   "artificial intelligence computer technology future",
                      "fantasy": "magic wizard dragon elf sword kingdom spell quest mythical "
                                 "enchanted sorcery castle medieval adventure epic",
                      "horror": "monster terror fear dark macabre nightmare haunted evil creature scary "
                                "frightening terrifying supernatural gothic demon ghost",
                      "gothic": "gothic mysterious brooding decay mansion atmosphere darkness melancholy "
                                "victorian romantic gloomy shadow castle",
                      "vampire": "vampire dracula blood undead night immortal fangs nocturnal "
                                 "bloodsucker supernatural creature darkness",
                      "dystopian": "dystopia totalitarian oppressive control surveillance regime "
                                   "authoritarian government society future dark oppression",
                      "post-apocalyptic": "apocalypse wasteland survival ruins collapse destroyed civilization "
                                          "nuclear war desolate barren end world",
                      "mystery": "mystery clue puzzle enigma secret hidden unknown detective solve "
                                 "investigation riddle suspense poe",
                      "thriller": "suspense tension danger chase escape threat action fast-paced "
                                  "adrenaline excitement dramatic",
                      "detective": "detective investigation solve crime evidence clue case murder "
                                   "police inspector sleuth forensic",
                      "romance": "love romance relationship passion heart desire emotion "
                                 "intimate affection romantic feelings couple",
                      "war": "war battle soldier combat military conflict weapon army fight violence warfare",
                      "historical": "historical history period era century past antiquity "
                                    "ancient medieval renaissance baroque",
                      "victorian": "victorian 19th century england british empire queen victoria london industrial era",
                      "adventure": "adventure journey expedition explore quest travel discovery voyage heroic action",
                      "philosophical": "philosophy meaning existence truth ethics morality consciousness "
                                       "reality metaphysics existential thought"}


print("Building genre matcher...")
genre_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
genre_names = list(GENRE_DESCRIPTIONS.keys())
genre_tfidf_matrix = genre_vectorizer.fit_transform(list(GENRE_DESCRIPTIONS.values()))

def fetch_all_discogs_styles(cache_file=STYLES_CACHE_FILE, force_refresh=False):
    # Fetch all music styles from Discogs API with caching.
    if not force_refresh and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                print(f"Loaded {len(cache_data['styles'])} styles from cache\n")
                return cache_data['styles'], cache_data['popularity']
        except Exception as e:
            print(f"Cache error: {e}, fetching fresh data...")

    print("Fetching music styles from Discogs API (first run only)...")
    print("This may take a few minutes...\n")

    style_counts = {}
    decades = [1960, 1970, 1980, 1990, 2000, 2010, 2020]
    formats = ["Vinyl", "CD", "Cassette", "Digital"]

    for decade in decades:
        for format_type in formats:
            try:
                time.sleep(MIN_SLEEP_SECONDS)
                print(f"Sampling {decade}s {format_type} releases...")
                results = d.search(type="release", year=decade, format=format_type, per_page=100, page=1)
                _respect_rate_limit(results)

                for release in results:
                    try:
                        if hasattr(release, 'styles') and release.styles:
                            for style in release.styles:
                                style_counts[style] = style_counts.get(style, 0) + 1
                    except:
                        continue
            except Exception as e:
                print(f"Error sampling {decade}s {format_type}: {e}")
                continue

    print("Sampling master releases for additional coverage...")
    try:
        time.sleep(MIN_SLEEP_SECONDS)
        masters = d.search(type="master", per_page=100, page=1)
        _respect_rate_limit(masters)

        for master in masters:
            try:
                if hasattr(master, 'styles') and master.styles:
                    for style in master.styles:
                        style_counts[style] = style_counts.get(style, 0) + 1
            except:
                continue
    except Exception as e:
        print(f"Error sampling masters: {e}")

    if not style_counts:
        print("WARNING: No styles found, using fallback list")
        style_counts = {"Rock": 1000000, "Pop": 900000, "Electronic": 800000,
                        "Jazz": 700000, "Classical": 600000, "Metal": 500000,
                        "Funk / Soul": 400000, "Hip Hop": 350000, "Reggae": 300000,
                        "Blues": 250000, "Folk": 200000, "Experimental": 150000}

    sorted_styles = sorted(style_counts.items(), key=lambda x: x[1], reverse=True)
    style_names = [style for style, _ in sorted_styles]
    style_popularity = dict(sorted_styles)

    cache_data = {"styles": style_names,
                  "popularity": style_popularity,
                  "timestamp": time.time(),
                  "count": len(style_names)}

    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cached {len(style_names)} styles to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    print(f"Discovered {len(style_names)} unique styles")
    print(f"Top 10: {', '.join(style_names[:10])}\n")

    return style_names, style_popularity

DISCOGS_STYLE_NAMES, DISCOGS_STYLE_POPULARITY = fetch_all_discogs_styles()

print("Building style matcher...")
style_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
style_tfidf_matrix = style_vectorizer.fit_transform(DISCOGS_STYLE_NAMES)
print("Ready\n")

def build_book_corpus(chosen_books: List[Dict]) -> Tuple[List[str], List[str]]:
    # Extract and combine text from books.
    titles, corpus = [], []
    for book in chosen_books:
        title = str(book.get("Full Title", "Unknown"))
        synopsis = str(book.get("Synopsis", "") or "")
        subjects = str(book.get("Subjects", "") or "")
        titles.append(title)
        corpus.append(f"{title} {synopsis} {subjects}".strip())
    return titles, corpus

def pick_main_genres(global_keywords, max_genres=3, similarity_threshold=0.1):
    # Match keywords to literary genres using TF-IDF cosine similarity.
    print(f"\nMatching {len(global_keywords)} keywords to genres...")
    genre_scores = {}

    for kw, kw_score in global_keywords:
        try:
            kw_vector = genre_vectorizer.transform([kw])
            similarities = cosine_similarity(kw_vector, genre_tfidf_matrix)[0]

            for idx, sim_score in enumerate(similarities):
                if sim_score > similarity_threshold:
                    genre = genre_names[idx]
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append((kw, kw_score * sim_score, sim_score))
        except:
            k = kw.lower()
            for genre in genre_names:
                if genre in k or k in genre:
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append((kw, kw_score, 1.0))
                    break

    if not genre_scores:
        print("No genre matches found, using default")
        return ["fantasy"]

    genre_confidence = {g: sum(s for _, s, _ in m) for g, m in genre_scores.items()}
    sorted_genres = sorted(genre_confidence.items(), key=lambda x: x[1], reverse=True)

    print("Top genre matches:")
    for genre, confidence in sorted_genres[:max_genres]:
        print(f"  {genre}: {confidence:.2f}")

    return [genre for genre, _ in sorted_genres[:max_genres]]

def map_genre_to_music_styles(genre: str, max_styles=3):
    # Map literary genre to music styles using hybrid scoring (70% semantic, 30% popularity).
    genre_desc = GENRE_DESCRIPTIONS.get(genre, genre)
    genre_vector = style_vectorizer.transform([genre_desc])
    similarities = cosine_similarity(genre_vector, style_tfidf_matrix)[0]

    weighted_scores = []
    for idx, sim in enumerate(similarities):
        if sim > 0.05:
            style = DISCOGS_STYLE_NAMES[idx]
            popularity = DISCOGS_STYLE_POPULARITY.get(style, 1)
            popularity_weight = math.log10(popularity + 1) / 7.0
            combined_score = (sim * 0.7) + (popularity_weight * 0.3)
            weighted_scores.append((style, sim, combined_score))

    weighted_scores.sort(key=lambda x: x[2], reverse=True)
    candidates = weighted_scores[:max_styles * 2]

    if not candidates:
        return []

    print(f"  Validating music styles for '{genre}'...")
    validated_styles = []

    for style, sim, combined in candidates:
        if len(validated_styles) >= max_styles:
            break
        try:
            time.sleep(MIN_SLEEP_SECONDS)
            test_results = d.search(type="release", style=style, per_page=1, page=1)
            _respect_rate_limit(test_results)

            if hasattr(test_results, 'page') and len(test_results.page(1)) > 0:
                validated_styles.append(style)
                print(f"    ✓ {style}")
        except:
            continue

    return validated_styles or [style for style, _, _ in weighted_scores[:max_styles]]

def _tracks_from_releases(results, genre, max_tracks=200):
    # Extract track metadata from Discogs releases.
    tracks = []
    for release in results:
        if len(tracks) >= max_tracks:
            break
        try:
            full_release = d.release(release.id)
            _respect_rate_limit(full_release)
            time.sleep(2.0)

            tracklist = full_release.tracklist or []
            if not tracklist or not tracklist[0].title:
                continue

            artist = full_release.artists[0].name.split(" (")[0] if full_release.artists else "Various"
            tracks.append({"title": tracklist[0].title,
                           "artist": artist,
                           "reason": genre,
                           "release": full_release.title,
                           "discogs_url": f"https://www.discogs.com/release/{release.id}",
                           "release_id": release.id})

        except:
            continue
    return tracks

def search_discogs_tracks_for_genre(genre: str, limit: int = 5):
    # Search Discogs for tracks matching genre.
    print(f"\nSearching Discogs for '{genre}'...")
    matched_styles = map_genre_to_music_styles(genre, max_styles=3)
    tracks = []

    for style in matched_styles:
        if tracks:
            break
        print(f"  Querying style: {style}")
        time.sleep(2.0)
        results = d.search(type="release", style=style, per_page=limit, page=1)
        _respect_rate_limit(results)
        tracks = _tracks_from_releases(results, genre, limit)

    if not tracks:
        print(f"  Trying fallback search...")
        time.sleep(2.0)
        results = d.search(type="release", title=genre, per_page=limit, page=1)
        _respect_rate_limit(results)
        tracks = _tracks_from_releases(results, genre, limit)

    print(f"  Found {len(tracks)} tracks")
    return tracks

def build_playlist_from_commonalities(global_keywords, total_tracks: int = 15):
    # Build a playlist from extracted keywords.
    genres = pick_main_genres(global_keywords, max_genres=3)
    per_theme = max(1, total_tracks // len(genres))

    print(f"\nBuilding {total_tracks}-track playlist...")
    playlist = []
    for i, genre in enumerate(genres):
        print(f"\nGenre {i + 1}/{len(genres)}: {genre}")
        tracks = search_discogs_tracks_for_genre(genre, limit=per_theme)
        playlist.extend(tracks)
        if i < len(genres) - 1:
            time.sleep(2.0)

    return playlist[:total_tracks]

def book_commonalities(chosen_books: List[Dict]) -> None:
    # Main workflow: analyse books and generate playlist.
    print("\n" + "="*60)
    print("ANALYSING BOOKS")
    print("="*60)

    titles, corpus = build_book_corpus(chosen_books)
    if len(corpus) < 2:
        print("Need at least 2 books")
        return

    print(f"\nSelected books:")
    for title in titles:
        print(f"  • {title}")

    print("\nExtracting themes with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, ngram_range=(1, 2), max_features=200)
    tfidf = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    global_scores = tfidf.sum(axis=0).A1
    top_idx = global_scores.argsort()[::-1][:15]
    global_keywords = [(feature_names[i], round(global_scores[i], 3)) for i in top_idx]

    print(f"\nTop themes detected:")
    for kw, score in global_keywords[:5]:
        print(f"  • {kw} ({score})")

    print("\nComputing book similarity...")
    sim = cosine_similarity(tfidf, tfidf)
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            print(f"  {titles[i]} ↔ {titles[j]}: {sim[i, j]:.2f}")

    playlist = build_playlist_from_commonalities(global_keywords, total_tracks=15)

    print("\n" + "="*60)
    print("FINAL PLAYLIST")
    print("="*60)
    print(f"Books: {', '.join(titles)}")
    print(f"Themes: {', '.join([kw for kw, _ in global_keywords[:5]])}")
    print("="*60)
    if playlist:
        for idx, track in enumerate(playlist, 1):
            print(f"{idx:2d}. {track['title']} – {track['artist']}")
            print(f"    Theme: {track['reason']} | Album: {track['release']}")
            print(f"    {track['discogs_url']}")
    else:
        print("No tracks found")
    print("="*60)

def main():
    try:
        print("\nAvailable books:")
        for idx, row in books_file.iterrows():
            print(f"{idx + 1}. {row['Full Title']}")

        books_num = int(input("\nHow many books? "))
        chosen_rows = []
        for i in range(books_num):
            while True:
                try:
                    choice = int(input(f"Select book #{i + 1} (1-{len(books_file)}): "))
                    if 1 <= choice <= len(books_file):
                        chosen_rows.append(books_file.iloc[choice - 1].to_dict())
                        break
                    else:
                        print("Out of range")
                except ValueError:
                    print("Enter a number")

        book_commonalities(chosen_rows)

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()


