import os
import urllib.request
import zipfile


dataPath = os.path.join(os.path.dirname(__file__), '..', 'data')
#ο φάκελος υπάρχει ήδη, οπότε η γραμμή διατηρείται για λόγους φορητότητας
os.makedirs(dataPath, exist_ok=True)


dataUrl = 'http://files.grouplens.org/datasets/movielens/ml-32m.zip'
zipPath = os.path.join(dataPath, 'ml-32m.zip')


print('Κατέβασμα της Βάσης Δεδομένων')
urllib.request.urlretrieve(dataUrl, zipPath)
print('Unzipping...')
with zipfile.ZipFile(zipPath, 'r') as z:
    z.extractall(dataPath)


print('Η διαδικασία ολοκληρώθηκε. Περιεχόμενα φακέλου: ', os.listdir(dataPath))


