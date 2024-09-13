testdocs:
	python prep_docs_doctests.py
	python -m pytest index.c.pytest.txt
	python -m pytest index.cl.pytest.txt

shortspeedtests:
	python gs_run.py --force True 
	python gs_parse.py
	rm -r -f gs_lattice.debug
	python nd_run.py --force True
	python nd_parse.py
	rm -r -f nd_lattice.debug

speedtestgs_lattice:
	python gs_run.py -q lattice -t full -m 14 -k 9 -s 1 -r 1 -p 1 -d 2 -f True
	python gs_parse.py -q lattice -t full -c cool -x k -y k -z k

speedtestgs_dnb2:
	python gs_run.py --qrproblem digital_net_base_2 --tag full --np2 13 --dp2 14 --gsnp2 2 --gsdp2 2 --skip 0 --runs 1 --platform 1 --device 2 --force True
	python gs_parse.py -q digital_net_base_2 -t full -c cool -x k -y k -z k

speedtestnd_lattice:
	python nd_run.py -q lattice -t full -g nd -m 16 -k 9 -s 1 -r 1 -p 1 -d 2 -f True
	python nd_parse.py -q lattice -t full -m 7 -c cool -x k -y k -z k

speedtestnd_dnb2:
	python nd_run.py -q digital_net_base_2 -t full -g nd -m 13 -k 14 -s 1 -r 1 -p 1 -d 2 -f True
	python nd_parse.py -q digital_net_base_2 -t full -m 7 -c cool -x k -y k -z k

mkdocsserve:
	mkdocs serve

exportcondaenv:
	conda env export --no-builds | tail -r | tail -n +2 | tail -r > myenv.yml
