sudo apt-get install python3-pip
sudo apt-get install python-dev
sudo apt-get install libxml2-dev libxslt-dev
sudo apt-get install libjpeg-dev zlib1g-dev libpng12-dev
curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3

pip install virtualenv
virtualenv -p /usr/bin/python3.10 venv
source venv/bin/activate
pip install Cython
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/ru_core_news_lg-3.3.0/ru_core_news_lg-3.3.0.tar.gz
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0.tar.gz

mkdir data
