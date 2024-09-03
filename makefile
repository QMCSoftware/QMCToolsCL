doctests:
	@rm -rf .pytest_cache
	@cat tests/cl_head.txt tests/tests.txt > tests/cl_tests.txt 
	@cat tests/c_head.txt tests/tests.txt > tests/c_tests.txt
	pytest tests/c_tests.txt
	pytest tests/cl_tests.txt