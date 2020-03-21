make clean:
	pyclean .
	find . -name "*.py[c|o]" -o -name __pycache__ -exec rm -rf {} +
	rm -rf tests/reports lidtk.egg-info .tox/ .cache/
