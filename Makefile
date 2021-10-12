

clean-install:
	pip uninstall flare
	pip uninstall synthobs
	pip install . -r requirements.txt

install:
	pip install . -r requirements.txt
