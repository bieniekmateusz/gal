import pandas as pd
import csv
import re
import time
from rdkit import Chem
import rdkit.Chem.Descriptors
from rdkit.Chem import Draw, AllChem
from selenium import webdriver
from bs4 import BeautifulSoup
from pathlib import Path


# url_template = "https://new.enaminestore.com/catalog/Z1898262832?cat=REALDB"

def get_1mg_price(enamine_id, browser=None):
	template = "https://new.enaminestore.com/catalog/%s?cat=REALDB"
	if browser is None:
		browser = webdriver.Chrome()

	browser.get(template % enamine_id)
	time.sleep(3)

	soup = BeautifulSoup(browser.page_source, "html")
	s = soup.find('select')
	if s is None:
		return "not available"
	mg1 = list(s.children)[0]
	if mg1.text.startswith('1 mg'):
		return "found"


chromium = webdriver.Chrome()

for sdf in Path(".").glob("*sdf"):
	print("sdf", sdf)
	for mol in Chem.SDMolSupplier(str(sdf)):
		enamine_id = mol.GetProp("enamine_id")

		price = get_1mg_price(enamine_id, browser=chromium)
		print(enamine_id, " is ", price)
		
