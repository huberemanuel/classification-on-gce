install:
	pip3 install -r requirements.txt
	pip3 install -e .

train:
	python3 -m classifier.train --n_epochs=2

test:
	pytest