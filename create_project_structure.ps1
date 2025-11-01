# 1️ Go to your base directory (adjust path as needed)
cd "C:\Users\crypt\Desktop\my fiu classes\Data Base Administration REPO\concurrent_web_server"

# 2️ Create the folder structure
mkdir llm_recommender, static, tests, reports, config, logs

# 3️ Create main entry file
New-Item -ItemType File -Path "server.py"

# 4️ llm_recommender package
New-Item -ItemType File -Path "llm_recommender\__init__.py"
New-Item -ItemType File -Path "llm_recommender\recommender.py"
New-Item -ItemType File -Path "llm_recommender\model_utils.py"

# 5️ static web files
New-Item -ItemType File -Path "static\index.html"
New-Item -ItemType File -Path "static\style.css"
New-Item -ItemType File -Path "static\script.js"

# 6️ tests
New-Item -ItemType File -Path "tests\test_concurrency.py"
New-Item -ItemType File -Path "tests\test_api.py"
New-Item -ItemType File -Path "tests\test_recommender.py"

# 7️ reports
New-Item -ItemType File -Path "reports\proposal.md"
New-Item -ItemType File -Path "reports\implementation_report.md"
New-Item -ItemType File -Path "reports\testing_report.md"

# 8️ config
New-Item -ItemType File -Path "config\__init__.py"
New-Item -ItemType File -Path "config\settings.py"

# 9️⃣ logs
New-Item -ItemType File -Path "logs\app.log"

# 10️ (optional) verify structure visually
tree /f
