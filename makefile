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

speedtestgs_fft:
	python gs_run.py --qrproblem fft --tag full --np2 0 --dp2 21 --gsminnp2 0 --gsnp2 0 --gsmindp2 5 --gsdp2 8 --skip 0 --runs 1 --platform 1 --device 2 --force True
	python gs_parse.py --qrproblem fft --tag full --cmap cool --colorleq1 k --colorgt1 k --colorperf k

speedtestgs_fwht:
	python gs_run.py --qrproblem fwht --tag full --np2 0 --dp2 21 --gsminnp2 0 --gsnp2 0 --gsmindp2 5 --gsdp2 8 --skip 0 --runs 1 --platform 1 --device 2 --force True
	python gs_parse.py --qrproblem fwht --tag full --cmap cool --colorleq1 k --colorgt1 k --colorperf k

speedtestgs_lattice:
	python gs_run.py --qrproblem lat_gen_gray --tag full --np2 16 --dp2 9 --gsminnp2 14 --gsnp2 16 --gsmindp2 8 --gsdp2 9 --skip 0 --runs 1 --platform 1 --device 2 --force True
	python gs_parse.py --qrproblem lat_gen_gray --tag full --cmap cool --colorleq1 k --colorgt1 k --colorperf k

speedtestgs_dnb2:
	python gs_run.py --qrproblem digital_net_base_2 --tag full --np2 14 --dp2 14 --gsminnp2 1 --gsnp2 12 --gsmindp2 5 --gsdp2 14 --skip 0 --runs 1 --platform 1 --device 2 --force True
	python gs_parse.py --qrproblem digital_net_base_2 --tag full --cmap cool --colorleq1 k --colorgt1 k --colorperf k

speedtestnd_lattice:
	python nd_run.py --qrproblem lattice --tag full --gsscheme customlattice --np2 19 --dp2 9 --skip 1 --runs 1 --platform 1 --device 2 --force True
	python nd_parse.py --qrproblem lattice --tag full --minnp2 1 --mindp2 1 --cmap cool --colorleq1 k --colorgt1 k --colorperf k

speedtestnd_dnb2:
	python nd_run.py --qrproblem digital_net_base_2 --tag full --gsscheme customdnb2 --np2 14 --dp2 14 --skip 1 --runs 1 --platform 1 --device 2 --force True
	python nd_parse.py --qrproblem digital_net_base_2 --tag full --minnp2 1 --mindp2 1 --cmap cool --colorleq1 k --colorgt1 k --colorperf k

mkdocsserve:
	mkdocs serve

exportcondaenv:
	conda env export --no-builds | tail -r | tail -n +2 | tail -r > myenv.yml

mydeviceinfo:
	python -c "import qmctoolscl; qmctoolscl.util.print_opencl_device_info()" > my_device_info.txt
