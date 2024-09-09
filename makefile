doctests:
	python prep_docs_doctests.py
	python -m pytest test.index.c.txt
	python -m pytest test.index.cl.txt

shortspeedtests:
	python gs_run.py --force True 
	python gs_parse.py
	rm -r -f gs_lattice.debug
	python nd_run.py --force True
	python nd_parse.py
	rm -r -f nd_lattice.debug

fullspeedtests:
	python gs_run.py -q lattice -t full -m 13 -k 9 -s 1 -r 1 -p 1 -d 2
	python gs_parse.py -q lattice -t full
	python nd_run.py -q lattice -t full -g nd -m 19 -k 9 -s 1 -r 1 -p 1 -d 2
	python nd_parse.py -q lattice -t full -m 7

mkdocsserve:
	mkdocs serve

exportcondaenv:
	conda env export --no-builds | tail -r | tail -n +2 | tail -r > myenv.yml
