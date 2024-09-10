testdocs:
	python prep_docs_doctests.py
	python -m pytest doctests.c.pytest.txt
	python -m pytest doctests.cl.pytest.txt
	python -m pytest examples.c.pytest.txt 
	python -m pytest examples.cl.pytest.txt

shortspeedtests:
	python gs_run.py --force True 
	python gs_parse.py
	rm -r -f gs_lattice.debug
	python nd_run.py --force True
	python nd_parse.py
	rm -r -f nd_lattice.debug

speedtestgs:
	python gs_run.py -q digital_net_base_2 -t full -m 12 -k 10 -s 1 -r 1 -p 1 -d 2 -f True
	python gs_parse.py -q digital_net_base_2 -t full -c cool -x k -y k -z k

speedtestnd:
	python nd_run.py -q digital_net_base_2 -t full -g custom -m 17 -k 9 -s 1 -r 1 -p 1 -d 2 -f True
	python nd_parse.py -q digital_net_base_2 -t full -m 7 -c cool -x k -y k -z k

mkdocsserve:
	mkdocs serve

exportcondaenv:
	conda env export --no-builds | tail -r | tail -n +2 | tail -r > myenv.yml
