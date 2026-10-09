"""Central configuration: random seed, skill dictionary, title rules.

To add a skill, add one line to DEFAULT_SKILLS:
    "canonical name": ["alias 1", "alias 2"],
Matching is case-insensitive and whole-token, so "java" will not match "javascript".
"""

RANDOM_SEED = 42

DEFAULT_SKILLS = {
    # Languages
    "python": ["python3"],
    "r": ["r programming", "r language", "rstudio"],
    "sql": ["mysql", "postgresql", "postgres", "t-sql", "pl/sql", "sqlite"],
    "java": ["core java", "java8"],
    "javascript": ["js", "es6", "ecmascript"],
    "typescript": ["ts"],
    "c++": ["cpp"],
    "c#": ["csharp"],
    "scala": [],
    "go": ["golang"],
    "bash": ["shell scripting", "shell script"],
    # Data / ML
    "machine learning": ["ml", "machine-learning"],
    "deep learning": ["dl", "neural networks", "neural network"],
    "natural language processing": ["nlp", "text mining"],
    "computer vision": ["image processing"],
    "statistics": ["statistical analysis", "statistical modeling", "statistical modelling"],
    "data analysis": ["data analytics", "analytical skills"],
    "data visualization": ["data visualisation", "dashboards", "dashboarding"],
    "feature engineering": [],
    "a/b testing": ["ab testing", "experimentation"],
    "time series": ["forecasting", "time-series"],
    "etl": ["data pipelines", "data pipeline", "elt"],
    "data modeling": ["data modelling", "data warehousing", "data warehouse"],
    # Libraries / frameworks
    "pandas": [],
    "numpy": [],
    "scikit-learn": ["sklearn", "scikit learn"],
    "tensorflow": [],
    "pytorch": ["torch"],
    "keras": [],
    "spark": ["pyspark", "apache spark"],
    "hadoop": [],
    "kafka": ["apache kafka"],
    "airflow": ["apache airflow"],
    "dbt": [],
    "react": ["reactjs", "react.js"],
    "angular": ["angularjs"],
    "node.js": ["nodejs", "node js"],
    "django": [],
    "flask": [],
    "fastapi": [],
    "spring boot": ["springboot", "spring"],
    "html": ["html5"],
    "css": ["css3"],
    "rest api": ["rest apis", "restful api", "restful apis", "restful"],
    # Tools / BI
    "tableau": [],
    "power bi": ["powerbi", "power-bi"],
    "excel": ["ms excel", "microsoft excel", "advanced excel"],
    "git": ["github", "version control"],
    "jira": [],
    "linux": ["unix"],
    # Cloud / DevOps
    "aws": ["amazon web services", "ec2"],
    "azure": ["microsoft azure"],
    "gcp": ["google cloud", "google cloud platform"],
    "cloud computing": ["cloud platforms", "cloud services"],
    "docker": ["containerization", "containers"],
    "kubernetes": ["k8s"],
    "ci/cd": ["cicd", "ci cd", "jenkins", "github actions"],
    "terraform": [],
    "mlops": ["ml ops", "model deployment"],
    # Databases
    "mongodb": ["mongo", "nosql"],
    "snowflake": [],
    "bigquery": ["big query"],
    # Soft / process
    "communication": ["communication skills"],
    "agile": ["scrum"],
}

# (regex, canonical role family) - first match wins
TITLE_RULES = [
    (r"data\s*scien", "Data Scientist"),
    (r"data\s*analy|business\s*analy|\bbi\s*(analyst|developer)", "Data Analyst"),
    (r"machine\s*learning|\bml\s*engineer|\bai\s*engineer|artificial\s*intelligence", "Machine Learning Engineer"),
    (r"data\s*engineer|\betl\b", "Data Engineer"),
    (r"devops|site\s*reliability|\bsre\b|cloud\s*engineer", "DevOps / Cloud Engineer"),
    (r"full[\s-]*stack", "Full-Stack Developer"),
    (r"front[\s-]*end|\bui\s*developer", "Frontend Developer"),
    (r"back[\s-]*end|software\s*(engineer|developer)|java\s*developer|python\s*developer", "Backend / Software Engineer"),
]

# Salary period -> annual multiplier (used only for salary normalization)
PERIOD_MULTIPLIERS = {"hour": 2080, "week": 52, "month": 12, "year": 1}
