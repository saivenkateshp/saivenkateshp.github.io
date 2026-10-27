"""Compute a 2-D 'skill embedding map' from Sai's résumé text.

Method (all deterministic, no external model):
  1. Corpus = résumé passages: v3 work-experience bullets + summary (weight 1.0),
     AIML project lines, where each project's header + bullets form one passage (weight 1.0),
     and the TECHNICAL SKILLS lines (weight 0.35, so grouping lists count less than real usage).
  2. Skill x passage matrix via alias matching.
  3. Weighted co-occurrence -> PPMI (positive pointwise mutual information).
  4. Cosine similarity between skills' PPMI vectors.
  5. t-SNE (precomputed distance = 1 - cosine) -> 2-D, normalised to [0,1].
  6. Top-3 nearest neighbours per skill from the cosine matrix.

Inputs (not committed, they contain personal contact details): v3.txt = résumé text,
aiml_projects.txt = the ACADEMIC PROJECTS section. Output: skillmap.json, inlined into index.html as MAP.
Run: pip install scikit-learn numpy && python embed_skills.py
"""
import json, re, numpy as np
from sklearn.manifold import TSNE

SKILLS = {  # label: (category, [regex aliases])
 "RAG":("GenAI & LLMs",[r"rag",r"retrieval-augmented"]),"Agentic AI":("GenAI & LLMs",[r"agentic",r"agent"]),
 "LangChain":("GenAI & LLMs",[r"langchain"]),"LangGraph":("GenAI & LLMs",[r"langgraph"]),"Google ADK":("GenAI & LLMs",[r"google adk",r"\badk\b"]),
 "Gemini API":("GenAI & LLMs",[r"gemini"]),"OpenAI API":("GenAI & LLMs",[r"openai"]),"Pinecone":("GenAI & LLMs",[r"pinecone",r"vector"]),
 "Prompt Engineering":("GenAI & LLMs",[r"prompt engineering"]),"NLP":("GenAI & LLMs",[r"\bnlp\b",r"natural language"]),"Guardrails":("GenAI & LLMs",[r"guardrails?"]),
 "PySpark":("Data Engineering",[r"pyspark",r"apache spark",r"\bspark\b"]),"Apache Kafka":("Data Engineering",[r"kafka"]),"Apache Airflow":("Data Engineering",[r"airflow"]),
 "Snowflake":("Data Engineering",[r"snowflake"]),"Databricks":("Data Engineering",[r"databricks"]),"ETL / ELT":("Data Engineering",[r"\betl\b",r"\belt\b",r"ingestion"]),
 "Star Schemas":("Data Engineering",[r"star schema",r"dimensional",r"slowly changing",r"data model"]),"BigQuery":("Data Engineering",[r"bigquery"]),
 "GCP":("Cloud",[r"\bgcp\b",r"google cloud"]),"GKE":("Cloud",[r"\bgke\b"]),"Vertex AI":("Cloud",[r"vertex ai"]),"Cloud Composer":("Cloud",[r"cloud composer"]),
 "Dataproc":("Cloud",[r"dataproc"]),"GCS":("Cloud",[r"\bgcs\b",r"cloud storage"]),"AWS":("Cloud",[r"\baws\b"]),
 "FastAPI":("Backend & APIs",[r"fastapi"]),"Spring Boot":("Backend & APIs",[r"spring boot"]),"Flask":("Backend & APIs",[r"flask"]),"GraphQL":("Backend & APIs",[r"graphql"]),
 "REST":("Backend & APIs",[r"\brest\b"]),"Microservices":("Backend & APIs",[r"microservices?"]),
 "TensorFlow":("ML & Search",[r"tensorflow"]),"Keras":("ML & Search",[r"keras"]),"OpenCV":("ML & Search",[r"opencv"]),"CNN":("ML & Search",[r"\bcnn\b",r"convolutional"]),
 "Elasticsearch":("ML & Search",[r"elastic ?search"]),"Apache Solr":("ML & Search",[r"\bsolr\b"]),"PageRank":("ML & Search",[r"pagerank",r"\bhits\b"]),
 "Python":("Languages",[r"python"]),"SQL":("Languages",[r"\bsql\b"]),"Java":("Languages",[r"\bjava\b"]),"JavaScript":("Languages",[r"javascript"]),
 "C++":("Languages",[r"c\+\+"]),"R":("Languages",[r"(?<![\w-])R(?![\w-])"]),
 "Docker":("DevOps & BI",[r"docker",r"containeri[sz]ed"]),"Kubernetes":("DevOps & BI",[r"kubernetes",r"\bgke\b"]),"Jenkins":("DevOps & BI",[r"jenkins"]),
 "CI/CD":("DevOps & BI",[r"ci/cd"]),"Grafana":("DevOps & BI",[r"grafana"]),"Prometheus":("DevOps & BI",[r"prometheus"]),"Tableau":("DevOps & BI",[r"tableau"]),
 "Power BI":("DevOps & BI",[r"power bi"]),"Kibana":("DevOps & BI",[r"kibana"]),
}
CASE_SENSITIVE = {"R"}

v3 = open("v3.txt").read().split("\n")
proj = open("aiml_projects.txt").read().split("\n")[1:]

docs = []  # (text, weight)
sk_start = v3.index("TECHNICAL SKILLS"); ed = v3.index("EDUCATION")
for line in v3[:sk_start]:
    if len(line) > 60: docs.append((line, 1.0))          # summary + work bullets
for line in v3[sk_start+1:ed]:
    docs.append((line, 0.35))                              # skills lines (weak signal)
cur = None
for line in proj:                                          # each project = one passage
    if "\t" in line:
        if cur: docs.append((cur, 1.0))
        cur = line.replace("\t", ". ")
    else:
        cur = (cur or "") + " " + line
if cur: docs.append((cur, 1.0))
# weak prior: one synthetic passage per site category (weight 0.2) so skills that only
# appear in the skills section still land near their own group instead of floating alone
CATDOCS = {}
for n,(c,_) in SKILLS.items(): CATDOCS.setdefault(c, []).append(n)
n_real = len(docs)

names = list(SKILLS)
M = np.zeros((len(names), len(docs)))
for i, n in enumerate(names):
    cat, al = SKILLS[n]
    flags = 0 if n in CASE_SENSITIVE else re.I
    rx = re.compile("|".join(al), flags)
    for j, (t, w) in enumerate(docs):
        if rx.search(t): M[i, j] = w

for c, members in CATDOCS.items():
    col = np.array([[0.2 if n in members else 0.0] for n in names])
    M = np.hstack([M, col])
missing = [n for i, n in enumerate(names) if M[i].sum() == 0]
assert not missing, f"skills never mentioned: {missing}"

C = M @ M.T                      # weighted co-occurrence
np.fill_diagonal(C, 0)
tot = C.sum(); row = C.sum(1, keepdims=True)
with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log((C * tot) / (row @ row.T))
P = np.nan_to_num(np.maximum(pmi, 0), posinf=0)
norm = np.linalg.norm(P, axis=1, keepdims=True); norm[norm == 0] = 1
U = P / norm
S = U @ U.T
np.fill_diagonal(S, 1)
D = np.clip(1 - S, 0, 2)

xy = TSNE(n_components=2, metric="precomputed", init="random", perplexity=9,
          random_state=7, max_iter=3000).fit_transform(D)
xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))

out = {}
for i, n in enumerate(names):
    order = [j for j in np.argsort(-S[i]) if j != i and S[i, j] > 0][:3]
    out[n] = {"x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4),
              "c": SKILLS[n][0], "nn": [names[j] for j in order],
              "uses": int((M[i, :n_real] >= 1).sum())}
json.dump(out, open("skillmap.json", "w"), ensure_ascii=False)
print(f"{len(docs)} passages, {len(names)} skills")
for n in ["RAG", "PySpark", "Kafka" if "Kafka" in out else "Apache Kafka", "Snowflake", "CNN", "FastAPI", "Tableau"]:
    print(f"{n:14s} nn={out[n]['nn']}  real-usage passages={out[n]['uses']}")
