install:
	pip3 install -r requirements.txt
	pip3 install -e .

train:
	python3 -m classifier.train --n_epochs=1

test:
	pytest

serve:
	uvicorn classifier.web.app:app 
