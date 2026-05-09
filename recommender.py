"""
DecodeLabs - Project 3 | Tech Stack Recommender
================================================

A simple, dependency-free career-path recommender that maps a user's skill set
to the closest matching job roles using a classic TF-IDF + cosine similarity
pipeline.

Pipeline Overview
-----------------
    1. Ingestion  - Load job roles and their skills from a CSV dataset and
                    capture the user's skills via the console.
    2. Scoring    - Convert every role and the user profile into a TF-IDF
                    weighted vector and compute cosine similarity.
    3. Sorting    - Rank roles by similarity score in descending order.
    4. Filtering  - Return the Top-N most relevant roles (with a Cold Start
                    fallback when the user provides no recognised skills).

Inputs
------
    data/raw_skills.csv : CSV with columns ``job_role`` and ``skills``,
                          where ``skills`` is a whitespace-separated list.

Usage
-----
    Run as a script::

        python recommender.py

    The user is prompted for at least three comma-separated skills and the
    Top-3 matching roles are printed with similarity scores.
"""

##========================
## Import Required Libraries
##========================
import csv
import math
from pathlib import Path

##===========================
## Project Working Directory
##===========================
PROJECT_DIR = Path(__file__).parent

##===============================================
## Data File Path
##===============================================
DATA_FILE = PROJECT_DIR / "data/raw_skills.csv"

##===============================================
## STEP 1: LOAD DATASET (Ingestion - Data Side)
##===============================================
def load_dataset(filepath):
    """Load job roles and their associated skills from a CSV file.

    The CSV is expected to have two columns:
        - ``job_role`` : the name/title of the role.
        - ``skills``   : a whitespace-separated string of skill tokens.

    Args:
        filepath (str | pathlib.Path): Path to the source CSV file.

    Returns:
        dict[str, list[str]]: A mapping of ``job_role`` to a list of skill
        tokens (e.g. ``{"Data Scientist": ["Python", "SQL", "Statistics"]}``).
    """
    dataset = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row['job_role']
            skills = row['skills'].split()
            dataset[role] = skills
    return dataset


##===============================================
## STEP 2: BUILD VOCABULARY
##===============================================
def build_vocabulary(dataset):
    """Build a sorted vocabulary of every unique skill across all roles.

    The vocabulary defines a fixed term ordering that is later used to align
    TF-IDF vectors so they are directly comparable across roles and the user
    profile.

    Args:
        dataset (dict[str, list[str]]): Mapping of role name to a list of
            skill tokens, as produced by :func:`load_dataset`.

    Returns:
        list[str]: Alphabetically sorted list of unique skill tokens.
    """
    vocab = set()
    for skills in dataset.values():
        vocab.update(skills)
    return sorted(list(vocab))


##===============================================
## STEP 3: COMPUTE TF (Term Frequency)
##===============================================
def compute_tf(skill_list):
    """Compute the Term Frequency (TF) for each skill in a single document.

    TF is defined as::

        TF(skill) = (occurrences of skill in document) / (total skills in document)

    A "document" here is the full skill list for one role (or the user).

    Args:
        skill_list (list[str]): Skills belonging to a single role or user.

    Returns:
        dict[str, float]: Mapping of each skill present in ``skill_list`` to
        its normalised term frequency (a value in ``[0, 1]``).
    """
    tf = {}
    total = len(skill_list)
    for skill in skill_list:
        tf[skill] = tf.get(skill, 0) + 1
    for skill in tf:
        tf[skill] /= total
    return tf


##===============================================
## STEP 4: COMPUTE IDF (Inverse Document Frequency)
##===============================================
def compute_idf(dataset):
    """Compute the Inverse Document Frequency (IDF) for every skill.

    IDF down-weights skills that appear in many roles (common, low-signal)
    and up-weights skills that appear in few roles (rare, high-signal)::

        IDF(skill) = log( total_roles / roles_containing_skill )

    Args:
        dataset (dict[str, list[str]]): Mapping of role name to its skill
            list. Each role is treated as a single "document".

    Returns:
        dict[str, float]: Mapping of each skill in the corpus to its IDF
        weight. Skills present in every role yield an IDF of ``0.0``.
    """
    total_docs = len(dataset)
    idf = {}
    all_skills = set()
    for skills in dataset.values():
        all_skills.update(skills)

    for skill in all_skills:
        docs_with_skill = sum(1 for skills in dataset.values() if skill in skills)
        idf[skill] = math.log(total_docs / docs_with_skill)
    return idf


##===============================================
## STEP 5: COMPUTE TF-IDF VECTOR
##===============================================
def compute_tfidf_vector(skill_list, idf, vocabulary):
    """Build a fixed-length TF-IDF weighted vector for a skill list.

    Each component of the returned vector corresponds to a term in
    ``vocabulary`` (in the same order), making the vectors of different
    documents directly comparable. Each component is computed as::

        vector[i] = TF(term_i) * IDF(term_i)

    Args:
        skill_list (list[str]): Skills for a single role or for the user.
        idf (dict[str, float]): IDF weights produced by :func:`compute_idf`.
        vocabulary (list[str]): Ordered list of all known skills, as
            produced by :func:`build_vocabulary`.

    Returns:
        list[float]: A vector of length ``len(vocabulary)`` with TF-IDF
        weights. Terms missing from the document or the IDF table are 0.
    """
    tf = compute_tf(skill_list)
    vector = []
    for term in vocabulary:
        tf_val = tf.get(term, 0)
        idf_val = idf.get(term, 0)
        vector.append(tf_val * idf_val)
    return vector


##===============================================
## STEP 6: COSINE SIMILARITY
##===============================================
def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two equal-length vectors.

    The metric measures the cosine of the angle between two vectors and is
    therefore independent of their magnitudes::

        cos(theta) = (A . B) / (||A|| * ||B||)

    Args:
        vec_a (list[float]): First vector (e.g. user TF-IDF profile).
        vec_b (list[float]): Second vector (e.g. role TF-IDF profile).

    Returns:
        float: Similarity in the range ``[0.0, 1.0]`` for non-negative
        vectors. Returns ``0.0`` if either vector has zero magnitude
        (the Cold Start fallback case).
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # Cold Start fallback
    return dot_product / (magnitude_a * magnitude_b)


##===============================================
## STEP 7: RECOMMEND (Scoring + Sorting + Filtering)
##===============================================
def recommend(user_skills, dataset, top_n=3):
    """Recommend the Top-N best-matching job roles for a user.

    Implements the full 4-step pipeline:
        1. Ingestion - user skills are accepted as input.
        2. Scoring   - the user profile and every role are converted to
                       TF-IDF vectors and compared via cosine similarity.
        3. Sorting   - roles are ranked by similarity in descending order.
        4. Filtering - the top ``top_n`` roles are printed to the console.

    A Cold Start fallback is triggered when none of the user's skills
    appear in the corpus vocabulary (the user vector is all zeros). In
    that case the function prints a list of trending/default roles instead
    of similarity-based matches.

    Args:
        user_skills (list[str]): Skills supplied by the user.
        dataset (dict[str, list[str]]): Mapping of role to its skill list.
        top_n (int, optional): Number of roles to display. Defaults to 3.

    Returns:
        None: Results are printed to standard output as a formatted report;
        nothing is returned to the caller.
    """
    vocabulary = build_vocabulary(dataset)
    idf = compute_idf(dataset)

    # User profile vector
    user_vector = compute_tfidf_vector(user_skills, idf, vocabulary)

    # Cold Start check
    if all(v == 0 for v in user_vector):
        print("\n⚠️  Cold Start Detected! No matching skills found in vocabulary.")
        print("💡  Tip: Showing globally trending roles instead:\n")
        for i, role in enumerate(list(dataset.keys())[:top_n], 1):
            print(f"   {i}. {role} (Trending Fallback)")
        return

    # Score each job role
    scores = {}
    for role, skills in dataset.items():
        role_vector = compute_tfidf_vector(skills, idf, vocabulary)
        scores[role] = cosine_similarity(user_vector, role_vector)

    # Sort descending
    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Filter Top-N
    top_results = sorted_roles[:top_n]

    print("\n" + "=" * 50)
    print("   🎯  TOP RECOMMENDED CAREER PATHS FOR YOU")
    print("=" * 50)
    for rank, (role, score) in enumerate(top_results, 1):
        bar = "█" * int(score * 20)
        print(f"\n  #{rank}  {role}")
        print(f"       Match Score : {score:.4f} ({score*100:.1f}%)")
        print(f"       [{bar:<20}]")
    print("\n" + "=" * 50)


##===============================================
## MAIN PROGRAM
##===============================================
def main():
    """Entry point for the command-line recommender experience.

    Steps performed:
        1. Loads the dataset from :data:`DATA_FILE`.
        2. Builds the skill vocabulary and reports its size.
        3. Prompts the user for at least three comma-separated skills,
           normalising spaces to underscores so multi-word skills match
           the dataset format (e.g. ``Machine Learning`` -> ``Machine_Learning``).
        4. Invokes :func:`recommend` to print the Top-3 matching roles.

    Returns:
        None: The function only performs I/O and has no return value.
    """
    print("=" * 50)
    print("  🤖  DecodeLabs - Tech Stack Recommender")
    print("      Project 3 | AI Recommendation Logic")
    print("=" * 50)

    # Load dataset
    dataset = load_dataset(DATA_FILE)
    print(f"\n✅  Dataset loaded: {len(dataset)} job roles found.")

    # Display available skills hint
    vocabulary = build_vocabulary(dataset)
    print(f"📚  Vocabulary size: {len(vocabulary)} unique skills.\n")

    # STEP 1: INGESTION - Collect user inputs (minimum 3)
    print("📝  Enter at least 3 skills you have (comma separated).")
    print("    Example: Python, Machine_Learning, SQL\n")

    while True:
        raw_input = input("Your skills: ").strip()
        user_skills = [s.strip().replace(" ", "_") for s in raw_input.split(",") if s.strip()]

        if len(user_skills) < 3:
            print("⚠️  Please enter at least 3 skills.\n")
        else:
            break

    print(f"\n✅  Skills captured: {user_skills}")

    # Run recommendation pipeline
    recommend(user_skills, dataset, top_n=3)

    print("\n💡  Tip: Try different skill combinations to explore more roles!")
    print("    Example skills: Docker, Kubernetes, AWS, Linux, CI_CD\n")


if __name__ == "__main__":
    main()
