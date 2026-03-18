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
GENRE_DESCRIPTIONS = {
    "science fiction": "space alien robot future technology planet android spaceship cosmos extraterrestrial sci-fi spacecraft galaxy interstellar",
    "cyberpunk": "cyber tech hacker dystopia corporate virtual digital network matrix artificial intelligence computer technology future",
    "fantasy": "magic wizard dragon elf sword kingdom spell quest mythical enchanted sorcery castle medieval adventure epic",
    "horror": "monster terror fear dark macabre nightmare haunted evil creature scary frightening terrifying supernatural gothic demon ghost",
    "gothic": "gothic mysterious brooding decay mansion atmosphere darkness melancholy victorian romantic gloomy shadow castle",
    "vampire": "vampire dracula blood undead night immortal fangs nocturnal bloodsucker supernatural creature darkness",
    "dystopian": "dystopia totalitarian oppressive control surveillance regime authoritarian government society future dark oppression",
    "post-apocalyptic": "apocalypse wasteland survival ruins collapse destroyed civilization nuclear war desolate barren end world",
    "mystery": "mystery clue puzzle enigma secret hidden unknown detective solve investigation riddle suspense poe",
    "thriller": "suspense tension danger chase escape threat action fast-paced adrenaline excitement dramatic",
    "detective": "detective investigation solve crime evidence clue case murder police inspector sleuth forensic",
    "romance": "love romance relationship passion heart desire emotion intimate affection romantic feelings couple",
    "war": "war battle soldier combat military conflict weapon army fight violence warfare",
    "historical": "historical history period era century past antiquity ancient medieval renaissance baroque",
    "victorian": "victorian 19th century england british empire queen victoria london industrial era",
    "adventure": "adventure journey expedition explore quest travel discovery voyage heroic action",
    "philosophical": "philosophy meaning existence truth ethics morality consciousness reality metaphysics existential thought"
}

# Initialize TF-IDF vectorizer for smart genre matching
print("[INIT] Building smart genre matcher with TF-IDF...")
genre_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
genre_names = list(GENRE_DESCRIPTIONS.keys())
genre_texts = list(GENRE_DESCRIPTIONS.values())
genre_tfidf_matrix = genre_vectorizer.fit_transform(genre_texts)

# Book genre to music style mapping
BOOK_TOPIC_TO_DISCOGS_STYLE = {"science fiction": ["Space Rock", "Space-Age", "Synth-pop"],
                               "space opera": ["Space Rock", "Symphonic Rock", "Prog Rock"],
                               "cyberpunk": ["Industrial", "Synthwave", "Electro"],
                               "fantasy": ["Power Metal", "Symphonic Rock", "Folk Metal"],
                               "epic": ["Symphonic Metal", "Power Metal", "Prog Rock"],
                               "mythology": ["Folk Metal", "Pagan", "Symphonic Rock"],
                               "medieval": ["Medieval", "Folk", "Neo-Classical"],
                               "horror": ["Horror Rock", "Horrorcore", "Darkwave"],
                               "gothic": ["Goth Rock", "Gothic Metal", "Darkwave"],
                               "occult": ["Occult", "Black Metal", "Dark Ambient"],
                               "vampire": ["Gothic Metal", "Goth Rock", "Darkwave"],
                               "dystopian": ["Industrial", "Industrial Metal", "EBM"],
                               "post-apocalyptic": ["Industrial Metal", "Doom Metal", "Sludge Metal"],
                               "political": ["Political", "Punk", "Hardcore"],
                               "mystery": ["Jazz", "Score", "Soundtrack"],
                               "thriller": ["Soundtrack", "Score", "Dark Jazz"],
                               "detective": ["Jazz", "Cool Jazz", "Lounge"],
                               "noir": ["Jazz", "Dark Jazz", "Lounge"],
                               "historical": ["Classical", "Baroque", "Opera"],
                               "war": ["Military", "Marches", "Symphonic Metal"],
                               "victorian": ["Baroque", "Classical", "Opera"],
                               "ancient": ["Classical", "Medieval", "Renaissance"],
                               "adventure": ["Symphonic Rock", "Score", "Soundtrack"],
                               "pirate": ["Sea Shanties", "Folk Rock", "Punk"],
                               "western": ["Country", "Folk", "Americana"],
                               "romance": ["Romantic", "Ballad", "Soft Rock"],
                               "heartbreak": ["Emo", "Indie Rock", "Ballad"],
                               "spiritual": ["New Age", "Ambient", "Drone"],
                               "religious": ["Religious", "Gospel", "Choral"],
                               "mystical": ["Ethereal", "Dark Ambient", "Drone"],
                               "literary": ["Art Rock", "Prog Rock", "Chamber Music"],
                               "satire": ["Parody", "Comedy", "Punk"],
                               "coming of age": ["Indie Rock", "Emo", "Pop Punk"],
                               "psychological": ["Experimental", "Abstract", "Avant-garde"],
                               "surreal": ["Psychedelic", "Experimental", "Abstract"],
                               "philosophical": ["Prog Rock", "Art Rock", "Avant-garde Jazz"]}

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
    """Use TF-IDF cosine similarity to match keywords to genres intelligently."""
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
            for genre in BOOK_TOPIC_TO_DISCOGS_STYLE.keys():
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

def search_discogs_tracks_for_genre(genre: str, limit: int = 5):
    # Search Discogs for tracks by music style.
    print(f"\n[DISCOGS] Searching '{genre}'...")
    style = BOOK_TOPIC_TO_DISCOGS_STYLE.get(genre, [None])[0]
    tracks = []

    if style:
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
