# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:26:29 2020

@author: sebas
"""



import requests
from bs4 import BeautifulSoup
url = 'http://webcache.googleusercontent.com/search?q=cache:fDWaiMV6MRYJ:mlr.cs.umass.edu/ml/datasets.html%3Fformat%3D%26task%3D%26att%3D%26area%3Dcomp%26numAtt%3D%26numIns%3D%26type%3D%26sort%3DinstUp%26view%3Dtable+&cd=1&hl=es-419&ct=clnk&gl=ar'

# Lets use the requests get method to fetch the data from url

response = requests.get(url)
# lets check the status
status = response.status_code
print(status) # 200 means the fetching was successful



import requests
import json
url = 'http://webcache.googleusercontent.com/search?q=cache:fDWaiMV6MRYJ:mlr.cs.umass.edu/ml/datasets.html%3Fformat%3D%26task%3D%26att%3D%26area%3Dcomp%26numAtt%3D%26numIns%3D%26type%3D%26sort%3DinstUp%26view%3Dtable+&cd=1&hl=es-419&ct=clnk&gl=ar'

response = requests.get(url)
content = response.content # we get all the content from the website
soup = BeautifulSoup(content, 'html.parser') # beautiful soup will give a chance to parse
print(soup.title) # <title>UCI Machine Learning Repository: Data Sets</title>
print(soup.title.get_text()) # UCI Machine Learning Repository: Data Sets
print(soup.body) # gives the whole page on the website
print(response.status_code)

tables = soup.find_all('table', {'cellpadding':'3'})
# We are targeting the table with cellpadding attribute with the value of 3
# We can select using id, class or HTML tag , for more information check the beautifulsoup doc
table = tables[0] # the result is a list, we are taking out data from it
dct = {}
flag=0
for td in table.find('tr').find_all('td'):
        if flag == 0:
            key = td.text
            flag =1
        if not(any(char.isdigit() for char in td.text)):
            key = td.text
        else:
            value= td.text
            dct[key]= value
for i in dct.keys():
  print(i)