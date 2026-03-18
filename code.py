import os
import time
import pandas as pd
import discogs_client
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration
DISCOGS_USER_TOKEN = os.getenv("DISCOGS_USER_TOKEN", "TwhKkMPvpWPwyZTsRCoPHHcabLZJJYddnKxmrXne")
RATE_LIMIT_BUFFER = 2
MIN_SLEEP_SECONDS = 1.0

# Initialize
print("[INIT] Loading books and Discogs client...")
books_file = pd.read_excel('Backup_Books.xlsx')
d = discogs_client.Client("BookPlaylistApp/1.0", user_token=DISCOGS_USER_TOKEN)
print(f"[INIT] Loaded {len(books_file)} books\n")

def _respect_rate_limit(response):
    # Handle Discogs API rate limiting.
    headers = getattr(response, 'headers', {}) or {}
    try:
        limit = int(headers.get('X-Discogs-Ratelimit', 0))
        remaining = int(headers.get('X-Discogs-Ratelimit-Remaining', 0))
    except (TypeError, ValueError):
        time.sleep(MIN_SLEEP_SECONDS)
        return

    if limit == 0 and remaining == 0:
        time.sleep(MIN_SLEEP_SECONDS)
    elif remaining <= RATE_LIMIT_BUFFER:
        print("[RATE] Near limit; sleeping 60 seconds...")
        time.sleep(60)
    else:
        time.sleep(MIN_SLEEP_SECONDS)

# Genre descriptions for smart semantic matching using TF-IDF
GENRE_DESCRIPTIONS = {"science fiction": "space alien robot future technology planet android "
                                         "spaceship cosmos extraterrestrial sci-fi spacecraft galaxy interstellar",
                      "cyberpunk": "cyber tech hacker dystopia corporate virtual digital network matrix "
                                   "artificial intelligence computer technology future",
                      "fantasy": "magic wizard dragon elf sword kingdom spell quest mythical "
                                 "enchanted sorcery castle medieval adventure epic",
                      "horror": "monster terror fear dark macabre nightmare haunted evil "
                                "creature scary frightening terrifying supernatural gothic demon ghost",
                      "gothic": "gothic mysterious brooding decay mansion atmosphere "
                                "darkness melancholy victorian romantic gloomy shadow castle",
                      "vampire": "vampire dracula blood undead night immortal fangs "
                                 "nocturnal bloodsucker supernatural creature darkness",
                      "dystopian": "dystopia totalitarian oppressive control surveillance regime "
                                   "authoritarian government society future dark oppression",
                      "post-apocalyptic": "apocalypse wasteland survival ruins collapse "
                                          "destroyed civilization nuclear war desolate barren end world",
                      "mystery": "mystery clue puzzle enigma secret hidden unknown "
                                 "detective solve investigation riddle suspense poe",
                      "thriller": "suspense tension danger chase escape threat action "
                                  "fast-paced adrenaline excitement dramatic",
                      "detective": "detective investigation solve crime evidence clue "
                                   "case murder police inspector sleuth forensic",
                      "romance": "love romance relationship passion heart desire "
                                 "emotion intimate affection romantic feelings couple",
                      "war": "war battle soldier combat military conflict weapon army fight violence warfare",
                      "historical": "historical history period era century past antiquity "
                                    "ancient medieval renaissance baroque",
                      "victorian": "victorian 19th century england british empire queen victoria london industrial era",
                      "adventure": "adventure journey expedition explore quest travel discovery voyage heroic action",
                      "philosophical": "philosophy meaning existence truth ethics morality "
                                       "consciousness reality metaphysics existential thought"}


# Initialize TF-IDF vectorizer for smart genre matching
print("[INIT] Building smart genre matcher with TF-IDF...")
genre_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
genre_names = list(GENRE_DESCRIPTIONS.keys())
genre_texts = list(GENRE_DESCRIPTIONS.values())
genre_tfidf_matrix = genre_vectorizer.fit_transform(genre_texts)

# Top Discogs styles with popularity weights (release count from Discogs database)
# Using top 100 most popular styles for better API results
DISCOGS_STYLES = [
    ("Pop Rock", 1042142), ("House", 813585), ("Experimental", 717705), ("Punk", 672562),
    ("Alternative Rock", 614477), ("Synth-pop", 597472), ("Techno", 571029), ("Indie Rock", 520209),
    ("Ambient", 515334), ("Hardcore", 492900), ("Disco", 489021), ("Folk", 485692),
    ("Country", 438996), ("Hard Rock", 427196), ("Electro", 392993), ("Rock & Roll", 374018),
    ("Romantic", 346223), ("Trance", 338347), ("Heavy Metal", 333457), ("Psychedelic Rock", 328506),
    ("Soundtrack", 309170), ("Folk Rock", 307722), ("Downtempo", 301363), ("Noise", 290934),
    ("Prog Rock", 280582), ("Funk", 271145), ("Classic Rock", 266524), ("Black Metal", 265272),
    ("Blues Rock", 234625), ("New Wave", 227497), ("Industrial", 226871), ("Classical", 222847),
    ("Death Metal", 222009), ("Drum n Bass", 209110), ("Soft Rock", 194414), ("Garage Rock", 185112),
    ("Abstract", 182680), ("Gospel", 175742), ("Baroque", 157974), ("Acoustic", 156558),
    ("Thrash", 154127), ("Modern", 153749), ("Swing", 147928), ("Indie Pop", 142276),
    ("Drone", 133289), ("Dub", 133188), ("Opera", 120698), ("IDM", 110574),
    ("Breakbeat", 130767), ("Post-Punk", 103222), ("Dark Ambient", 102363), ("Art Rock", 102070),
    ("Fusion", 100340), ("Reggae", 100291), ("Doom Metal", 97592), ("Religious", 97192),
    ("Avantgarde", 91796), ("Score", 85993), ("Rockabilly", 85538), ("Comedy", 85016),
    ("Jazz-Funk", 85001), ("Lo-Fi", 84202), ("Grindcore", 79863), ("Leftfield", 77134),
    ("Ska", 76401), ("Post Rock", 76395), ("Spoken Word", 75483), ("Psy-Trance", 74756),
    ("Power Pop", 74080), ("Dubstep", 73344), ("Glam", 73295), ("New Age", 73090),
    ("Hip Hop", 70052), ("Goth Rock", 69597), ("Modern Classical", 69016), ("Jazz-Rock", 67057),
    ("Emo", 66307), ("Choral", 65543), ("Free Improvisation", 64715), ("Musical", 62099),
    ("Trip Hop", 61134), ("Stoner Rock", 60022), ("EBM", 58677), ("Shoegaze", 58472),
    ("Jungle", 55560), ("Synthwave", 54361), ("Hard Bop", 54118), ("Tango", 53686),
    ("Free Jazz", 53327), ("Trap", 53264), ("Darkwave", 51484), ("Cool Jazz", 50077),
    ("Vaporwave", 49738), ("Bluegrass", 49594), ("Metalcore", 49421), ("Laïkó", 49071),
    ("UK Garage", 47020), ("Novelty", 45324), ("Smooth Jazz", 45216), ("Grunge", 44678),
    ("Progressive Metal", 44646), ("Flamenco", 44066), ("AOR", 42756), ("Nu Metal", 40776),
    ("Boom Bap", 40668), ("Symphonic Rock", 38661), ("Power Metal", 35401), ("Space Rock", 34024),
    ("Bossa Nova", 33746), ("Krautrock", 33601), ("Post-Hardcore", 33439), ("Speed Metal", 32109),
    ("Neo-Classical", 31861), ("Breakcore", 31725), ("Avant-garde Jazz", 31232), ("Power Electronics", 30425),
    ("Horrorcore", 12332), ("Horror Rock", 5669), ("Gothic Metal", 18796)
]

# Initialize TF-IDF vectorizer for genre-to-music-style matching using style names
print("[INIT] Building smart genre-to-music-style matcher with TF-IDF...")
style_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
style_names = [style for style, _ in DISCOGS_STYLES]
style_popularity = {style: count for style, count in DISCOGS_STYLES}
# Use style names themselves as the corpus for semantic matching
style_tfidf_matrix = style_vectorizer.fit_transform(style_names)

def build_book_corpus(chosen_books: List[Dict]) -> Tuple[List[str], List[str]]:
    # Build text corpus from books (title + synopsis + subjects).
    print("[CORPUS] Building corpus...")
    titles, corpus = [], []
    for idx, book in enumerate(chosen_books, start=1):
        title = str(book.get("Full Title", "Unknown"))
        synopsis = str(book.get("Synopsis", "") or "")
        subjects = str(book.get("Subjects", "") or "")
        text = f"{title} {synopsis} {subjects}".strip()
        print(f"  Book #{idx}: {title}")
        titles.append(title)
        corpus.append(text)
    return titles, corpus

def pick_main_genres(global_keywords, max_genres=3, similarity_threshold=0.1):
    # Use TF-IDF cosine similarity to match keywords to genres intelligently.
    print(f"\n[GENRES] Analyzing {len(global_keywords)} keywords with semantic matching...")
    genre_scores = {}

    for kw, kw_score in global_keywords:
        # Transform keyword using the same vectorizer
        try:
            kw_vector = genre_vectorizer.transform([kw])

            # Compute cosine similarity with all genres
            similarities = cosine_similarity(kw_vector, genre_tfidf_matrix)[0]

            # Find genres above the threshold
            for idx, sim_score in enumerate(similarities):
                if sim_score > similarity_threshold:
                    genre = genre_names[idx]
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    # Weight by both keyword importance and semantic similarity
                    weighted_score = kw_score * sim_score
                    genre_scores[genre].append((kw, weighted_score, sim_score))
                    print(f"  '{kw}' -> {genre} (similarity: {sim_score:.2f})")
        except:
            # Fallback: exact match
            k = kw.lower()
            for genre in genre_names:
                if genre in k or k in genre:
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append((kw, kw_score, 1.0))
                    print(f"  '{kw}' -> {genre} (exact match)")
                    break

    if not genre_scores:
        print("  WARNING: No matches! Using fallback 'fantasy'")
        return ["fantasy"]

    # Rank by confidence (sum of weighted scores)
    genre_confidence = {g: sum(s for _, s, _ in m) for g, m in genre_scores.items()}
    sorted_genres = sorted(genre_confidence.items(), key=lambda x: x[1], reverse=True)

    print("\n[GENRES] Top matches:")
    for genre, confidence in sorted_genres[:max_genres]:
        keywords = [f"{kw} ({sim:.2f})" for kw, _, sim in genre_scores[genre]]
        print(f"  {genre}: {confidence:.2f}")
        print(f"    Keywords: {', '.join(keywords)}")

    return [genre for genre, _ in sorted_genres[:max_genres]]

def _tracks_from_releases(results, genre, max_tracks=200):
    # Extract tracks from Discogs releases.
    tracks = []
    for idx, release in enumerate(results, start=1):
        if len(tracks) >= max_tracks:
            break
        try:
            release_id = release.id
            full_release = d.release(release_id)
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
                           "discogs_url": f"https://www.discogs.com/release/{release_id}",
                           "release_id": release_id})

            print(f"  ✓ {tracklist[0].title} – {artist}")
        except Exception as e:
            continue
    return tracks

def map_genre_to_music_styles(genre: str, max_styles=3, similarity_threshold=0.05):
    """Use TF-IDF cosine similarity + popularity weighting to map book genre to Discogs music styles."""
    import math
    try:
        # Get genre description from GENRE_DESCRIPTIONS
        genre_desc = GENRE_DESCRIPTIONS.get(genre, genre)
        genre_vector = style_vectorizer.transform([genre_desc])

        # Compute cosine similarity with all music styles
        similarities = cosine_similarity(genre_vector, style_tfidf_matrix)[0]

        # Weight by both semantic similarity AND popularity (log scale to avoid overwhelming)
        weighted_scores = []
        for idx, sim in enumerate(similarities):
            if sim > similarity_threshold:
                style = style_names[idx]
                popularity = style_popularity[style]
                # Combine semantic similarity with log-scaled popularity
                popularity_weight = math.log10(popularity + 1) / 7.0  # Normalize to ~0-1 range
                combined_score = (sim * 0.7) + (popularity_weight * 0.3)  # 70% semantic, 30% popularity
                weighted_scores.append((style, sim, combined_score))

        # Sort by combined score
        weighted_scores.sort(key=lambda x: x[2], reverse=True)

        matched_styles = [style for style, _, _ in weighted_scores[:max_styles]]

        if matched_styles:
            print(f"  Genre '{genre}' → Styles:")
            for style, sim, combined in weighted_scores[:max_styles]:
                print(f"    {style} (semantic: {sim:.2f}, combined: {combined:.2f})")

        return matched_styles
    except Exception as e:
        print(f"  Warning: Style matching failed ({e}), trying fallback")
        return []

def search_discogs_tracks_for_genre(genre: str, limit: int = 5):
    # Search Discogs for tracks by music style using smart TF-IDF matching.
    print(f"\n[DISCOGS] Searching '{genre}'...")
    matched_styles = map_genre_to_music_styles(genre, max_styles=3)
    tracks = []

    # Try each matched style
    for style in matched_styles:
        if tracks:
            break
        print(f"  Using style: {style}")
        time.sleep(2.0)
        results = d.search(type="release", style=style, per_page=limit, page=1)
        _respect_rate_limit(results)
        tracks = _tracks_from_releases(results, genre, limit)

    # Fallback to title search
    if not tracks:
        print(f"  Fallback: title search")
        time.sleep(2.0)
        results = d.search(type="release", title=genre, per_page=limit, page=1)
        _respect_rate_limit(results)
        tracks = _tracks_from_releases(results, genre, limit)

    print(f"  Found {len(tracks)} tracks")
    return tracks

def build_playlist_from_commonalities(global_keywords, total_tracks: int = 15):
    # Build a playlist from keywords.
    genres = pick_main_genres(global_keywords, max_genres=3)
    per_theme = max(1, total_tracks // len(genres))

    print(f"\n[PLAYLIST] Building {total_tracks}-track playlist from {len(genres)} genres...")
    playlist = []
    for i, genre in enumerate(genres):
        print(f"\n[PLAYLIST] Genre {i + 1}/{len(genres)}: {genre}")
        tracks = search_discogs_tracks_for_genre(genre, limit=per_theme)
        playlist.extend(tracks)
        if i < len(genres) - 1:
            time.sleep(2.0)

    return playlist[:total_tracks]

def book_commonalities(chosen_books: List[Dict]) -> None:
    # Main analysis: Extract themes from books and generate a playlist.
    print("\n" + "="*60)
    print("ANALYZING BOOKS")
    print("="*60)

    # Step 1: Build corpus
    titles, corpus = build_book_corpus(chosen_books)
    if len(corpus) < 2:
        print("Need at least 2 books.")
        return

    # Step 2: TF-IDF to extract keywords
    print("\n[NLP] Extracting themes with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, ngram_range=(1, 2), max_features=200)
    tfidf = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # Step 3: Top keywords
    global_scores = tfidf.sum(axis=0).A1
    top_idx = global_scores.argsort()[::-1][:15]
    global_keywords = [(feature_names[i], round(global_scores[i], 3)) for i in top_idx]

    print(f"\nTop themes detected:")
    for kw, score in global_keywords[:5]:
        print(f"  • {kw} ({score})")

    # Step 4: Similarity
    print("\n[NLP] Computing book similarity...")
    sim = cosine_similarity(tfidf, tfidf)
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            print(f"  {titles[i]} ↔ {titles[j]}: {sim[i, j]:.2f}")

    # Step 5: Build playlist
    playlist = build_playlist_from_commonalities(global_keywords, total_tracks=15)

    # Step 6: Display results
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
        print("No tracks found.")
    print("="*60)

def eval():
    # compare the songs to each other for the evaluation
    pass


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
                        print("Out of range.")
                except ValueError:
                    print("Enter a number.")

        book_commonalities(chosen_rows)

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
