doctests:
	python prep_docs_doctests.py
	python -m pytest test.index.c.txt
	python -m pytest test.index.cl.txt

mkdocsserve:
	mkdocs serve
