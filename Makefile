install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
		
kaggle_download:
	cd data/ &&\
	kaggle datasets download uciml/sms-spam-collection-dataset

extract:
	unzip data/sms-spam-collection-dataset.zip -d s3_transfer
	
s3_upload:
	aws s3 sync data/s3_transfer s3://dsba-6190-project3-spark

test:
	#python -m pytest -vv --cov=myrepolib tests/*.py
	#python -m pytest --nbval notebook.ipynb

lint:
	#pylint --disable=R,C main.py

all: install lint test