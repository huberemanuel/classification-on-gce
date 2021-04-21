install:
	pip3 install -r requirements.txt
	pip3 install -e .

train:
	python3 -m classifier.train --n_epochs=1

test:
	pytest

serve:
	uvicorn classifier.web.app:app 

docker-serve:
	docker run -v ~/.config/gcloud:/root/.config/gcloud -p 8084:8084 classifier_web