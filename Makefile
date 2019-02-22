

clean-install:
	pip uninstall FLARE
	pip uninstall SynthObs
	pip install . -r requirements.txt

install:
	pip install . -r requirements.txt