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
    """Load job roles and their skills from CSV."""
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
    """Create a unified vocabulary of all unique skills."""
    vocab = set()
    for skills in dataset.values():
        vocab.update(skills)
    return sorted(list(vocab))


##===============================================
## STEP 3: COMPUTE TF (Term Frequency)
##===============================================
def compute_tf(skill_list):
    """TF = count of skill / total skills in role."""
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
    """IDF = log(Total Docs / Docs containing term)."""
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
    """Create a TF-IDF weighted vector for a skill list."""
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
    """cos(θ) = (A · B) / (||A|| * ||B||)"""
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
    """
    Full 4-Step Pipeline:
    1. Ingestion   - user skills captured
    2. Scoring     - cosine similarity per role
    3. Sorting     - descending order by score
    4. Filtering   - return Top-N results
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
