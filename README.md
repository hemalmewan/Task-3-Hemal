# DecodeLabs - Tech Stack Recommender

**Project 3 | AI Recommendation Logic**

A content-based career path recommender that suggests the most suitable tech job roles based on a user's skill set. The system uses **TF-IDF vectorization** and **cosine similarity** to score and rank job roles against the user's profile.

---

## Overview

Given a list of skills entered by the user, the program:

1. **Ingests** the user's skills.
2. **Scores** every job role in the dataset using cosine similarity over TF-IDF vectors.
3. **Sorts** the roles in descending order of match score.
4. **Filters** and displays the Top-N recommended career paths.

A **Cold Start** fallback is included: if the user enters skills that do not exist in the vocabulary, the system shows globally trending roles instead.

---

## Project Structure

```
Project-3/
├── data/
│   └── raw_skills.csv              # Job roles and their associated skills
├── recommender.py                  # Main recommendation pipeline
├── Artificial Intelligence Project 3.pdf
└── README.md
```

---

## Dataset

The dataset [data/raw_skills.csv](data/raw_skills.csv) contains job roles mapped to their required skills.

**Format:**

| job_role        | skills                                            |
|-----------------|---------------------------------------------------|
| Data Scientist  | Python SQL Machine_Learning Pandas NumPy ...      |
| ML Engineer     | Python TensorFlow PyTorch Docker Kubernetes ...   |
| Frontend Dev    | HTML CSS JavaScript React Vue.js TypeScript ...   |

> Multi-word skills use underscores (e.g. `Machine_Learning`, `REST_APIs`).

---

## How It Works

The pipeline in [recommender.py](recommender.py) follows a 7-step process:

| Step | Function                  | Purpose                                              |
|------|---------------------------|------------------------------------------------------|
| 1    | `load_dataset()`          | Load job roles & skills from CSV                     |
| 2    | `build_vocabulary()`      | Build a unified vocabulary of all unique skills      |
| 3    | `compute_tf()`            | Compute Term Frequency for each skill list           |
| 4    | `compute_idf()`           | Compute Inverse Document Frequency over the dataset  |
| 5    | `compute_tfidf_vector()`  | Build TF-IDF weighted vectors                        |
| 6    | `cosine_similarity()`     | Score similarity between user and role vectors       |
| 7    | `recommend()`             | Sort and filter Top-N recommendations                |

### Core Formulas

- **TF** = `count(skill) / total_skills_in_role`
- **IDF** = `log(total_docs / docs_containing_skill)`
- **TF-IDF** = `TF × IDF`
- **Cosine Similarity** = `(A · B) / (||A|| × ||B||)`

---

## Requirements

- Python 3.7+
- No external dependencies — uses only the Python standard library (`csv`, `math`, `pathlib`).

---

## Usage

From the project root, run:

```bash
python recommender.py
```

You will be prompted to enter at least 3 skills, comma-separated:

```
Your skills: Python, Machine_Learning, SQL
```

### Example Output

```
==================================================
   🎯  TOP RECOMMENDED CAREER PATHS FOR YOU
==================================================

  #1  Data Scientist
       Match Score : 0.7421 (74.2%)
       [██████████████      ]

  #2  ML Engineer
       Match Score : 0.6105 (61.1%)
       [████████████        ]

  #3  Data Analyst
       Match Score : 0.5832 (58.3%)
       [███████████         ]
==================================================
```

---

## Tips

- Use underscores for multi-word skills (e.g. `Deep_Learning`, `CI_CD`).
- Try varied skill combinations to discover roles you may not have considered.
- Example skills to explore: `Docker, Kubernetes, AWS, Linux, CI_CD`.

---

## Author

Built as part of the **DecodeLabs AI** learning track — Project 3.
