# Disney+ Cosine Similarity Recommender

An educational Python project that builds a **content-based recommendation system** from the supplied Disney+ titles CSV file. It uses metadata, TF-IDF and cosine similarity to find movies and TV shows that are similar to a title you choose.

## 1. Project files

```text
disney_cosine_recommender/
├── .vscode/
│   └── launch.json
├── disney_plus_titles.csv
├── disney_recommender.py
├── README.md
└── requirements.txt
```

The CSV is included, so the project can be moved to another folder or computer as a self-contained example.

## 2. Open and run it in Visual Studio Code

**Python version:** Python 3.10 or newer.

1. Extract the project folder if you downloaded it as a ZIP.
2. Open Visual Studio Code.
3. Choose **File → Open Folder...** and select `disney_cosine_recommender`.
4. Open the integrated terminal (**Terminal → New Terminal**).
5. Create a virtual environment:

   **Windows (PowerShell):**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

6. Install the required Python packages:
   ```bash
   python -m pip install -r requirements.txt
   ```

7. Run the program:
   ```bash
   python disney_recommender.py
   ```

   The program will ask you to enter a Disney+ title. Try a title that exists in the CSV, for example `Moana`, `Frozen`, or `The Lion King` (the exact available matches depend on the supplied dataset).

If the `python` command is not recognized on Windows, try `py` instead. In VS Code, you can also select the virtual environment as the Python interpreter using **Python: Select Interpreter** from the Command Palette.

## 3. Other ways to run it

Search for a particular title and show five recommendations:

```bash
python disney_recommender.py --title "Moana"
```

Show ten recommendations:

```bash
python disney_recommender.py --title "Moana" --top 10
```

Use a CSV stored somewhere else:

```bash
python disney_recommender.py --data "path/to/disney_plus_titles.csv"
```

See all supported options:

```bash
python disney_recommender.py --help
```

## 4. What does cosine similarity do?

A recommendation system needs a way to compare two titles. Cosine similarity does this by comparing the **angle between two numerical vectors**.

For feature vectors `A` and `B`:

\[
\operatorname{cosine\ similarity}(A,B)
= \frac{A \cdot B}{\|A\|\,\|B\|}
\]

- `A · B` is the dot product of the two vectors.
- `||A||` is the length (Euclidean norm) of vector A.
- `||B||` is the length (Euclidean norm) of vector B.

For this dataset, feature values are non-negative, so scores will normally be between 0 and 1:

| Score | General interpretation |
|---:|---|
| Near 1.0 | The titles have strongly aligned feature vectors. |
| Near 0.5 | The titles share some feature information. |
| Near 0.0 | The titles have little or no overlap in the features used. |

The exact meaning of a score depends on the features and weights in the model. A score of `0.80` does **not** mean an 80% probability that a viewer will like the title. It means the two encoded metadata vectors are closely aligned under this model.

## 5. How a title becomes a vector

The CSV has descriptive columns, not numerical vectors. The script converts them into numbers before comparing titles.

### A. Multi-label metadata

The script uses `MultiLabelBinarizer` for genres, director, cast and country. For example, imagine a tiny catalogue containing these genres:

| Title | Genres |
|---|---|
| A | Animation, Family |
| B | Comedy, Family |
| C | Documentary |

The genre part of the vectors would look like this:

| Title | Animation | Comedy | Documentary | Family |
|---|---:|---:|---:|---:|
| A | 1 | 0 | 0 | 1 |
| B | 0 | 1 | 0 | 1 |
| C | 0 | 0 | 1 | 0 |

The shared `Family` feature gives A and B an overlap. A and C do not share a genre feature in this small example.

A value of `1` means that a title has that label; `0` means that it does not. The same idea is used for directors, actors and countries. These groups are stored as sparse matrices because most titles contain only a small fraction of all possible labels.

### B. Single-value categories

The program also encodes the content type (`Movie` or `TV Show`), rating and release decade. The release year is grouped into a decade so that, for example, 2011 and 2018 belong to the same `2010s` group.

### C. Description text and TF-IDF

The plot description contains many words. `TfidfVectorizer` turns those words into numerical features.

TF-IDF stands for **Term Frequency–Inverse Document Frequency**:

- Term Frequency measures how much a term appears in one description.
- Inverse Document Frequency gives less weight to terms that occur in many descriptions and more weight to terms that are relatively distinctive.
- The vectorizer removes common English stop words and uses both single words and two-word phrases (`ngram_range=(1, 2)`).

This helps distinguish descriptions by their story-related vocabulary rather than simply counting every common word equally.

### D. Combining the features

The code joins the feature groups into one long vector per title. The `FEATURE_WEIGHTS` dictionary near the top of `disney_recommender.py` lets you experiment with how influential each group is.

For example, increasing the genre weight gives genre overlap more influence. Increasing the description weight makes the synopsis words more influential. The weights in this project are illustrative and should be tuned with evaluation data for a real product.

## 6. The recommendation process

1. You select a title.
2. The program finds that title's row in the combined feature matrix.
3. It reads the cosine similarity between the selected row and every other row.
4. It removes the selected title itself (which would otherwise have a score of 1.0 against itself).
5. It sorts the remaining titles from the highest score to the lowest.
6. It displays the top N titles.

The similarity matrix is computed once when the program starts. This is practical for the supplied dataset of 1,450 titles. For very large catalogues, a full all-pairs matrix would use much more time and memory; production systems normally use indexing, nearest-neighbour search or other scalable approaches.

## 7. Experiments to understand the model

Try these small experiments and compare the recommendations:

1. **Genre-first recommendations:** change `"genres": 3.0` to `5.0` in `FEATURE_WEIGHTS`.
2. **Story-first recommendations:** change `"description": 2.0` to `4.0`.
3. **Ignore cast:** change `"cast": 1.5` to `0.0`.
4. **Ignore release period:** change `"decade": 0.5` to `0.0`.
5. **Compare different titles:** run the program with several starting titles and see which recommendations repeat.

After each change, save the file and run the program again. Notice that the recommendations change because you have changed the representation of what counts as similar.

## 8. Limitations — what this model does not know

- It does not know what a particular user has watched, rated, liked or disliked.
- It does not learn from the behaviour of similar users. That would be a different approach called collaborative filtering.
- It treats the dataset's metadata as the basis for similarity. Incomplete or inaccurate metadata can affect the results.
- Shared actors or genres do not guarantee that two titles have the same tone, quality or audience appeal.
- The feature weights are manually chosen for teaching, not learned from user feedback.

A useful next project would be to compare these content-based recommendations with a collaborative-filtering approach using a dataset that contains user–item ratings or viewing events.
