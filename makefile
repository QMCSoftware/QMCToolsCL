doctests:
	python prep_docs_doctests.py
	python -m pytest test.index.c.txt
	python -m pytest test.index.cl.txt

shortspeedtests:
	mkdir -p shortspeedtests
	python gs_run.py --force True 
	python gs_parse.py
	python nd_run.py --force True
	python nd_parse.py

mkdocsserve:
	mkdocs serve

exportcondaenv:
	conda env export --no-builds | tail -r | tail -n +2 | tail -r > myenv.yml
