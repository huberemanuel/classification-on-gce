install:
	pip install -r requirements.txt
	pip install -e .

train:
	python -m classifier.train --n_epochs=2

test:
	pytest