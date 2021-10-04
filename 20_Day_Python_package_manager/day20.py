# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:13:33 2020

@author: sebas
"""

import webbrowser # web browser module to open websites

# list of urls: python
url_lists = [
    'http://www.python.org',
    'https://www.linkedin.com/in/asabeneh/',
    'https://twitter.com/Asabeneh',
    'https://twitter.com/Asabeneh',
]

# opens the above list of websites in a different tab
for url in url_lists:
    webbrowser.open_new_tab(url)
    
import requests # importing the request module

url = 'https://www.w3.org/TR/PNG/iso_8859-1.txt' # text from a website

response = requests.get(url) # opening a network and fetching a data
print(response)
print(response.status_code) # status code, success:200
print(response.headers)     # headers information
print(response.text) # gives all the text from the page

# Let's read from an api. API stands for Application Program Interface. It is a means to exchange structure data between servers primarily a json data. An example of an api:https://restcountries.eu/rest/v2/all. Let's read this API using requests module.
import requests
url = 'https://restcountries.eu/rest/v2/all'  # countries api
response = requests.get(url)  # opening a network and fetching a data
print(response) # response object
print(response.status_code)  # status code, success:200
countries = response.json()
print(countries[:1])  # we sliced only the first country, remove the slicing to see all countries

# #1
print(1)
import requests
url = 'http://www.gutenberg.org/files/1112/1112.txt'  # txt api
response = requests.get(url)  # opening a network and fetching a data
print(response) # response object
print(response.status_code)  # status code, success:200
text = response.text


def find_most_common_words(text,number):    
    # Python program to find the k most frequent words 
    # from data set 
    from collections import Counter
    # split() returns list of all the words in the string 
    split_it = text.split()
    # Pass the split_it list to instance of Counter class. 
    Counter = Counter(split_it)
    # most_common() produces k frequently encountered 
    # input values and their respective counts. 
    most_common_words = Counter.most_common(number) 
    return most_common_words


print(find_most_common_words(text, 10))

url = 'https://api.thecatapi.com/v1/breeds'  # json api
response = requests.get(url)  # opening a network and fetching a data
print(response) # response object
print(response.status_code)  # status code, success:200
cats = response.json()
b =[]
for i in cats:
    a = i['weight']['metric']
    b.append(list(a.split(" - ")))
print(b[3][0])
count1 =0
count2 =0
for i in range(len(b)):
    count1 += int(b[i][0])
    count2 += int(b[i][1])
    
print("Average cat in metric: "+"%.2f"%(count1/len(b))+" - "+"%.2f"%(count2/len(b)))

import requests
url = 'https://restcountries.eu/rest/v2/all'  # countries api
response = requests.get(url)  # opening a network and fetching a data
print(response) # response object
print(response.status_code)  # status code, success:200
countries = response.json()
a = []
b = []
for i in countries:
   a.append((i['name']))
   if i['area'] is None:
       b.append(("%.2f"%0))
   else :
       b.append("%.2f"%i['area'])   
b, a = zip(*sorted(zip(b, a)))
sort_countries = dict(zip(a, b))
sort_countries = sorted(sort_countries.items(), key=lambda x: float(x[1]), reverse=True)
for i in range(10):
    print("The country number %d"%(i+1)+" is",sort_countries[i][0]," with area: %s"%sort_countries[i][1])