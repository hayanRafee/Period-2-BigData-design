"""
DISNEY+ CONTENT-BASED RECOMMENDATION SYSTEM
===========================================

This educational project demonstrates how cosine similarity can recommend
Disney+ movies and TV shows that have metadata in common.

What the program does:
1. Loads the Disney+ CSV file.
2. Cleans missing and inconsistent metadata.
3. Converts metadata into numerical feature vectors.
4. Uses cosine similarity to compare every title with every other title.
5. Finds and displays the most similar titles when you enter a title.

IMPORTANT IDEA:
A computer cannot calculate cosine similarity directly from words such as
"Animation", "Family", or "Mickey Mouse". We first have to represent each
title as a numerical vector. This process is called feature engineering.

This is a CONTENT-BASED recommender: recommendations are based on properties
of the titles, not on users' viewing histories or ratings from other users.
"""

# -----------------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


# -----------------------------------------------------------------------------
# 2. PROJECT SETTINGS
# -----------------------------------------------------------------------------

# Path(__file__) means "the folder in which this Python file is saved".
# This makes the default CSV path work even when VS Code is opened from a
# different working directory, as long as the CSV is next to this script.
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "disney_plus_titles.csv"

# These weights control how much each group of information contributes to
# the final title vector. They are deliberately easy to find and experiment
# with: try changing a number and see how the recommendations change.
#
# A larger weight makes that feature group more influential in the final
# cosine similarity calculation. These are teaching choices, not universal
# values. A real streaming service would test and tune weights using data.
FEATURE_WEIGHTS = {
    "genres": 3.0,       # Genre is often a strong clue about similar content.
    "description": 2.0,  # The plot / synopsis gives us detailed text signals.
    "cast": 1.5,         # Shared actors can indicate related titles.
    "director": 1.5,     # Shared directors can indicate a similar style.
    "type": 1.0,         # Movie and TV Show are different content types.
    "country": 0.5,      # Production country is a weaker signal in this demo.
    "rating": 0.5,       # Age rating is useful, but not a genre by itself.
    "decade": 0.5,       # Similar release periods can be a small extra signal.
}

# Columns that must exist for the project to work. If a different CSV is used,
# the program will give a clear error instead of failing later in the code.
REQUIRED_COLUMNS = {
    "show_id",
    "type",
    "title",
    "director",
    "cast",
    "country",
    "release_year",
    "rating",
    "listed_in",
    "description",
}


# -----------------------------------------------------------------------------
# 3. THE RECOMMENDER CLASS
# -----------------------------------------------------------------------------

class DisneyPlusRecommender:
    """Build and query a content-based recommender for the Disney+ data."""

    def __init__(self, csv_path: Path | str):
        """Load the data and build all feature vectors and similarity scores.

        The work in this initializer is done once when the recommender starts.
        After that, recommending a title is quick because the similarity
        matrix has already been calculated.
        """
        self.csv_path = Path(csv_path)

        print("Loading the Disney+ dataset...")
        self.data = self._load_and_clean_data(self.csv_path)

        print(f"Loaded {len(self.data):,} titles.")
        print("Converting title information into numerical features...")
        self.feature_matrix = self._build_feature_matrix()

        print("Calculating cosine similarity between all titles...")
        # cosine_similarity compares every row-vector with every other
        # row-vector. The result is a square matrix:
        #   rows    = titles used as the starting title
        #   columns = titles that could be recommended
        #
        # For this dataset (1,450 titles), the matrix is small enough to keep
        # in memory. For a catalogue with millions of titles, calculating and
        # storing every possible pair would be too expensive; approximate
        # nearest-neighbour methods or on-demand calculations would be better.
        self.similarity_matrix = cosine_similarity( #[REF]
            self.feature_matrix,
            self.feature_matrix,
            dense_output=True,
        )

        print("The recommender is ready!\n")

    # -------------------------------------------------------------------------
    # 3A. LOAD AND CLEAN THE DATA
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_and_clean_data(csv_path: Path) -> pd.DataFrame:
        """Read the CSV and make fields safe to use in the model.

        The supplied dataset has missing values in fields such as director,
        cast, country, date_added and rating. Missing information is normal in
        real datasets. We should handle it deliberately rather than allowing
        the word 'nan' (Not a Number) to become a fake director or actor.
        """
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Could not find the CSV file:\n{csv_path}\n\n"
                "Place disney_plus_titles.csv next to this Python file, or run "
                "the program with --data followed by the CSV file path."
            )

        # read_csv turns the CSV file into a pandas DataFrame (a table).
        df = pd.read_csv(csv_path)

        # Check that the columns required by this program really exist.
        missing_columns = REQUIRED_COLUMNS.difference(df.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                "The CSV file is missing required column(s): " + missing_text
            )

        # Work on a copy so that cleaning does not change the original object.
        df = df.copy()

        # Remove accidental spaces around titles. We retain the original
        # capitalization because it looks nicer when displaying results.
        df["title"] = df["title"].fillna("Unknown title").astype(str).str.strip()

        # The model does not need date_added or duration, so we do not use them.
        # For the fields we do use, fill missing values with an empty string.
        # An empty string becomes an empty feature list later on; it does not
        # pretend that all missing directors/cast members are the same person.
        for column in ["director", "cast", "country", "listed_in", "description"]:
            df[column] = df[column].fillna("").astype(str)

        # Unknown is an explicit category for a missing age rating or type.
        # This keeps the row usable and makes the missing value visible.
        df["rating"] = df["rating"].fillna("Unknown").astype(str).str.strip()
        df["type"] = df["type"].fillna("Unknown").astype(str).str.strip()

        # Convert release_year to a number. If a year is absent or invalid,
        # use NaN temporarily; it will be given the category "unknown" below.
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

        # A decade groups years into broader periods (for example, 1990s or
        # 2010s). This is less strict than asking two titles to have exactly
        # the same release year.
        def year_to_decade(year: float) -> str:
            if pd.isna(year):
                return "unknown"
            decade_start = (int(year) // 10) * 10
            return f"{decade_start}s"

        df["release_decade"] = df["release_year"].apply(year_to_decade)

        # The original dataset has valid release years. This fallback keeps
        # the display code robust if another compatible CSV has a blank year.
        df["release_year_display"] = df["release_year"].fillna(0).astype(int)
        df.loc[df["release_year"].isna(), "release_year_display"] = -1

        return df.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 3B. TURN COMMA-SEPARATED DATA INTO LISTS OF LABELS
    # -------------------------------------------------------------------------

    @staticmethod
    def _split_labels(value: str) -> List[str]:
        """Split fields such as 'Comedy, Family' into clean label tokens.

        MultiLabelBinarizer (used below) needs a list of labels for each title.
        We use casefold() so that 'Comedy' and 'COMEDY' are treated equally.
        Extra spaces are removed too.
        """
        if not value:
            return []

        labels = []
        for part in str(value).split(","):
            cleaned = " ".join(part.strip().casefold().split())
            if cleaned:
                labels.append(cleaned)

        return labels

    # -------------------------------------------------------------------------
    # 3C. FEATURE ENGINEERING: WORDS AND METADATA -> NUMBERS
    # -------------------------------------------------------------------------

    def _encode_multiple_choice_column(self, column: str):
        """One-hot encode a comma-separated metadata column.

        Example:
            Title A genres: ['animation', 'family']
            Title B genres: ['comedy', 'family']

        The encoded features are conceptually:
            Title A: animation=1, comedy=0, family=1
            Title B: animation=0, comedy=1, family=1

        A shared label gives the titles a shared feature. MultiLabelBinarizer
        creates a sparse matrix, which stores non-zero values efficiently.
        """
        label_lists = self.data[column].apply(self._split_labels)

        # sparse_output=True avoids creating a large table full of zeros.
        encoder = MultiLabelBinarizer(sparse_output=True)
        encoded_matrix = encoder.fit_transform(label_lists)

        return encoded_matrix, encoder

    def _encode_single_category_column(self, values: Sequence[str]):
        """One-hot encode a column where each title has one category.

        MultiLabelBinarizer expects a list of labels for each row, so we wrap
        every single value in a one-item list. This lets us use one consistent
        encoding approach for genres, cast, director, type, rating and decade.
        """
        label_lists = [[str(value).casefold()] for value in values]
        encoder = MultiLabelBinarizer(sparse_output=True)
        encoded_matrix = encoder.fit_transform(label_lists)
        return encoded_matrix, encoder

    def _build_feature_matrix(self):
        """Create one combined numerical vector for every Disney+ title.

        There are two different ways we represent information here:

        1. Categorical / multi-label metadata becomes one-hot features.
           Examples: genres, actors, director, country, type and rating.

        2. Description text becomes TF-IDF features. TF-IDF gives more weight
           to words that are meaningful for a title and less weight to words
           that occur in almost every description.

        Finally, scipy.sparse.hstack joins the feature groups side by side.
        The result has one row per title, and many columns (features).
        """
        feature_blocks = []

        # Each CSV column below may contain several comma-separated values.
        multi_label_columns = {
            "genres": "listed_in",
            "director": "director",
            "cast": "cast",
            "country": "country",
        }

        # First: encode genres, director, cast and country.
        for feature_name, csv_column in multi_label_columns.items():
            matrix, encoder = self._encode_multiple_choice_column(csv_column)

            # Multiplication by the feature weight scales the importance of
            # this whole group before the final cosine calculation.
            matrix = matrix.multiply(FEATURE_WEIGHTS[feature_name])
            feature_blocks.append(matrix)

            # Keep the encoder objects. They are not required to generate
            # recommendations after training, but are useful to inspect when
            # learning how categories were converted into columns.
            setattr(self, f"{feature_name}_encoder", encoder)

            print(
                f"  {feature_name:12s}: "
                f"{matrix.shape[1]:5,d} numerical features"
            )

        # Second: encode one-value categories (type, rating and decade).
        single_category_columns = {
            "type": self.data["type"],
            "rating": self.data["rating"],
            "decade": self.data["release_decade"],
        }

        for feature_name, values in single_category_columns.items():
            matrix, encoder = self._encode_single_category_column(values)
            matrix = matrix.multiply(FEATURE_WEIGHTS[feature_name])
            feature_blocks.append(matrix)
            setattr(self, f"{feature_name}_encoder", encoder)

            print(
                f"  {feature_name:12s}: "
                f"{matrix.shape[1]:5,d} numerical features"
            )

        # Third: convert the description into TF-IDF values.
        #
        # TF-IDF means Term Frequency - Inverse Document Frequency:
        # - Term Frequency (TF): how often a word appears in this description.
        # - Inverse Document Frequency (IDF): how rare that word is across all
        #   descriptions. Very common words receive less importance.
        #
        # stop_words='english' removes common English words such as "the",
        # "and" and "is". They usually tell us little about a story.
        # ngram_range=(1, 2) uses single words and two-word phrases. For example,
        # "space adventure" can be a more useful clue than "space" alone.
        self.description_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10_000,
        )
        description_matrix = self.description_vectorizer.fit_transform(
            self.data["description"]
        )
        description_matrix = description_matrix.multiply(
            FEATURE_WEIGHTS["description"]
        )
        feature_blocks.append(description_matrix)

        print(
            f"  {'description':12s}: "
            f"{description_matrix.shape[1]:5,d} TF-IDF features"
        )

        # hstack joins all the feature groups horizontally.
        # If there are 1,450 titles, the final matrix has 1,450 rows.
        # Every column represents a possible piece of information, such as a
        # genre, actor, decade, or word/phrase in a description.
        combined_matrix = hstack(feature_blocks, format="csr")

        print(
            f"\nCombined feature matrix: "
            f"{combined_matrix.shape[0]:,} titles x "
            f"{combined_matrix.shape[1]:,} features"
        )

        return combined_matrix

    # -------------------------------------------------------------------------
    # 4. FIND A TITLE IN THE CATALOGUE
    # -------------------------------------------------------------------------

    def find_title_matches(self, query: str) -> List[int]:
        """Return row positions for exact or partial title matches.

        The search is case-insensitive. If no exact title is found, the
        program tries titles that start with the query, then titles that
        contain the query anywhere.
        """
        query = query.strip().casefold()
        if not query:
            return []

        normalized_titles = self.data["title"].str.casefold()

        # Exact title matches are best because they identify the requested
        # title without ambiguity in most cases.
        exact = np.flatnonzero((normalized_titles == query).to_numpy()).tolist()
        if exact:
            return exact

        # A starts-with match is often more helpful than a general substring.
        starts_with = np.flatnonzero(
            normalized_titles.str.startswith(query).to_numpy()
        ).tolist()
        if starts_with:
            return starts_with

        # Last resort: any title that contains the entered words.
        contains = np.flatnonzero(normalized_titles.str.contains(
            query, regex=False, na=False
        ).to_numpy()).tolist()
        return contains

    # -------------------------------------------------------------------------
    # 5. RECOMMEND SIMILAR TITLES WITH COSINE SIMILARITY
    # -------------------------------------------------------------------------

    def recommend(self, title_index: int, number_of_recommendations: int = 5):
        """Return the most similar titles to the title at title_index.

        Cosine similarity compares the angle between two vectors:

                    A . B
            ---------------------
            ||A|| multiplied by ||B||

        A and B are the feature vectors for two titles.
        A . B is their dot product, and ||A|| is the vector length (norm).

        Interpretation:
        - 1.0 means the vectors point in the same direction (very similar).
        - 0.0 means the vectors are perpendicular (no shared direction).
        - -1.0 means opposite directions.

        Because our metadata and TF-IDF values are non-negative, our scores
        will normally be between 0 and 1. A score of 0.75 means the feature
        vectors are quite aligned; it does NOT mean a 75% chance that someone
        will enjoy the recommendation.
        """
        if not 0 <= title_index < len(self.data):
            raise IndexError("The selected title index is outside the dataset.")

        if number_of_recommendations < 1:
            raise ValueError("The number of recommendations must be at least 1.")

        # Select the row of the similarity matrix for the chosen title.
        scores = self.similarity_matrix[title_index].copy()

        # A title is always 100% similar to itself. We must remove it so it
        # does not appear as its own recommendation.
        scores[title_index] = -1.0

        # argsort returns positions sorted from low to high. [::-1] reverses
        # the order so that the highest cosine similarity appears first.
        top_indices = np.argsort(scores)[::-1][:number_of_recommendations]

        # Select the useful display columns from the original DataFrame.
        result_columns = [
            "show_id",
            "title",
            "type",
            "release_year_display",
            "rating",
            "listed_in",
            "description",
        ]
        recommendations = self.data.iloc[top_indices][result_columns].copy()

        # Add the similarity score next to each recommendation. Keep it as a
        # number between 0 and 1 internally; the display function converts it
        # into a percentage for easier reading.
        recommendations.insert(0, "cosine_similarity", scores[top_indices])
        recommendations = recommendations.reset_index(drop=True)

        return recommendations

    # -------------------------------------------------------------------------
    # 6. DISPLAY SEARCH RESULTS AND RECOMMENDATIONS
    # -------------------------------------------------------------------------

    def display_title_matches(self, matching_indices: Sequence[int]) -> None:
        """Print matching titles so the user can choose the right one."""
        print("\nMatching titles:")
        for option_number, row_index in enumerate(matching_indices, start=1):
            row = self.data.iloc[row_index]
            year = row["release_year_display"]
            year_text = str(year) if year != -1 else "year unknown"
            print(
                f"  {option_number}. {row['title']} "
                f"({row['type']}, {year_text})"
            )

    def display_recommendations(
        self,
        selected_index: int,
        recommendations: pd.DataFrame,
    ) -> None:
        """Print the chosen title and its recommended similar titles."""
        selected = self.data.iloc[selected_index]
        selected_year = selected["release_year_display"]
        year_text = str(selected_year) if selected_year != -1 else "unknown year"

        print("\n" + "=" * 78)
        print("YOUR SELECTED TITLE")
        print("=" * 78)
        print(f"Title:       {selected['title']}")
        print(f"Type:        {selected['type']}")
        print(f"Release year:{' ' if selected_year != -1 else ' '}{year_text}")
        print(f"Rating:      {selected['rating']}")
        print(f"Genres:      {selected['listed_in'] or 'Not supplied'}")
        print(f"Description: {selected['description'] or 'Not supplied'}")

        print("\n" + "=" * 78)
        print("RECOMMENDATIONS (highest cosine similarity first)")
        print("=" * 78)

        if recommendations.empty:
            print("No recommendations were found.")
            return

        for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
            score_as_percentage = row["cosine_similarity"] * 100
            year = row["release_year_display"]
            year_text = str(year) if year != -1 else "unknown year"

            print(f"\n{rank}. {row['title']} ({year_text})")
            print(f"   Similarity: {score_as_percentage:.1f}%")
            print(f"   Type:       {row['type']}")
            print(f"   Rating:     {row['rating']}")
            print(f"   Genres:     {row['listed_in'] or 'Not supplied'}")
            print(f"   Description:{' ' if row['description'] else ' Not supplied'}")
            if row["description"]:
                # Keep the command-line output readable by shortening very
                # long plot descriptions. The original CSV is not changed.
                description = row["description"]
                if len(description) > 220:
                    description = description[:217].rstrip() + "..."
                print(f"               {description}")

        print("\nNote: similarity is a comparison of metadata, not a user rating.")


# -----------------------------------------------------------------------------
# 7. USER INTERFACE: INTERACTIVE MODE OR COMMAND-LINE MODE
# -----------------------------------------------------------------------------

def choose_match_interactively(
    recommender: DisneyPlusRecommender,
    matching_indices: Sequence[int],
) -> int | None:
    """Ask the user to choose a title when a search has multiple matches."""
    if len(matching_indices) == 1:
        return int(matching_indices[0])

    recommender.display_title_matches(matching_indices[:20])

    if len(matching_indices) > 20:
        print("Showing the first 20 matches. Refine your search if needed.")

    while True:
        answer = input(
            "\nEnter the number of the title you mean (or press Enter to cancel): "
        ).strip()

        if answer == "":
            return None

        if answer.isdigit():
            choice = int(answer)
            if 1 <= choice <= min(len(matching_indices), 20):
                return int(matching_indices[choice - 1])

        print("Please enter one of the displayed option numbers.")


def run_interactive_mode(recommender: DisneyPlusRecommender) -> None:
    """Let the user search for multiple titles until they choose to quit."""
    print("DISNEY+ COSINE-SIMILARITY RECOMMENDER")
    print("Type a movie or TV show title. Type 'q' to quit.")

    while True:
        query = input("\nWhich Disney+ title do you like? ").strip()

        if query.casefold() in {"q", "quit", "exit"}:
            print("Goodbye. Happy learning!")
            break

        matching_indices = recommender.find_title_matches(query)

        if not matching_indices:
            print(
                f"No title containing '{query}' was found. "
                "Check the spelling or try a shorter search."
            )
            continue

        selected_index = choose_match_interactively(
            recommender,
            matching_indices,
        )

        if selected_index is None:
            print("Search cancelled.")
            continue

        while True:
            amount_text = input(
                "How many recommendations would you like? [5]: "
            ).strip()
            if amount_text == "":
                number_to_show = 5
                break
            if amount_text.isdigit() and int(amount_text) > 0:
                number_to_show = int(amount_text)
                break
            print("Please enter a positive whole number, such as 5 or 10.")

        recommendations = recommender.recommend(
            selected_index,
            number_of_recommendations=number_to_show,
        )
        recommender.display_recommendations(selected_index, recommendations)


# -----------------------------------------------------------------------------
# 8. PROGRAM ENTRY POINT
# -----------------------------------------------------------------------------

def parse_arguments():
    """Read optional settings typed after the script name in the terminal."""
    parser = argparse.ArgumentParser(
        description="Recommend similar Disney+ titles using cosine similarity."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to disney_plus_titles.csv (default: file beside this script).",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title to search for. Without this option, interactive mode starts.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of recommendations to display (default: 5).",
    )
    return parser.parse_args()


def main() -> int:
    """Start the recommender and handle common errors gracefully."""
    args = parse_arguments()

    if args.top < 1:
        print("Error: --top must be at least 1.", file=sys.stderr)
        return 2

    try:
        recommender = DisneyPlusRecommender(args.data)

        # When no title is supplied, start the interactive search loop.
        if not args.title:
            run_interactive_mode(recommender)
            return 0

        # If --title was supplied, search the dataset for it.
        matching_indices = recommender.find_title_matches(args.title)
        if not matching_indices:
            print(f"No Disney+ title matched: {args.title}")
            print("Try running without --title to search interactively.")
            return 1

        selected_index = choose_match_interactively(
            recommender,
            matching_indices,
        )
        if selected_index is None:
            print("No title selected.")
            return 1

        recommendations = recommender.recommend(
            selected_index,
            number_of_recommendations=args.top,
        )
        recommender.display_recommendations(selected_index, recommendations)
        return 0

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


# Python sets __name__ to "__main__" when this file is run directly.
# This guard prevents the program from automatically starting if the file is
# imported into another Python file or a notebook for experimentation.
if __name__ == "__main__":
    raise SystemExit(main())
