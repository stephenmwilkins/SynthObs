

clean-install:
	pip uninstall FLARE
	pip install . -r requirements.txt

install:
	pip install . -r requirements.txt