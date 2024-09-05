doctests:
	python prep_docs_doctests.py
	python -m pytest test.index.c.txt
	python -m pytest test.index.cl.txt

mkdocsserve:
	mkdocs serve

exportcondaenv:
	conda env export --no-builds | tail -r | tail -n +2 | tail -r > myenv.yml
