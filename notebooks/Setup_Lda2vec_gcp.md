# How to get lda2vec to run on gcp

- Follow https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52 for instance setup
- Configure GPUs
- Install cuda if not installed

- Install Anaconda
```
wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh
bash Anaconda3-4.0.0-Linux-x86_64.sh
source ~/.bashrc
```

- Create new conda virtual environment with python 3.6:
```
conda create -n myenv python=3.6
```

- Install requirements.txt verbatim:
```
six==1.12.0
pandas==0.21.1
numpy==1.16.2
scikit-learn==0.19.1
tensorflow-gpu==1.5.0
pyLDAvis==2.1.2 (pip)
Keras==2.1.4
tqdm==4.23.4
setuptools==40.1.0
```

- Install tensorflow==1.8.0 (this was the money maker, not listed in requirements.txt, https://github.com/tensorflow/tensorflow/issues/20778#issuecomment-410960014)
- install spacy
- install `en_core_web_sm`:
```
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
```
- install nltk

### Notes

Training takes forever but hey.. it works now which is amazing. I was about ready to jump off a bridge. Learned a lot about python, basically that versioning sucks and that python should be deprecated... or that you HAVE to use virtual environments or all of your shit will break. Honestly I'm super surprised that Python has such horrific backwards compatability issues.. kinda makes me wanna cry.





