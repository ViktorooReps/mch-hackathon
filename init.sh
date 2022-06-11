sudo apt-get install python3-pip
sudo apt-get install python-dev
sudo apt-get install libxml2-dev libxslt-dev
sudo apt-get install libjpeg-dev zlib1g-dev libpng12-dev
curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3

pip install virtualenv
virtualenv -p /usr/bin/python3.10 venv
source venv/bin/activate
pip install -r requirements.txt