doctests:
	python prep_docs_doctests.py
	pytest test.index.c.txt
	pytest test.index.cl.txt

mkdocsserve:
	mkdocs serve
