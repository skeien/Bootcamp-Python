# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:32:59 2020

@author: sebas
"""

f = open('reading_file_example.txt')
print(f) # <_io.TextIOWrapper name='./files/reading_file_example.txt' mode='r' encoding='UTF-8'>

f = open('reading_file_example.txt')
txt = f.read()
print(type(txt))
print(txt)
f.close()

f = open('reading_file_example.txt')
txt = f.read(15)
print(type(txt))
print(txt)
f.close()

#readline(): read only the first line
f = open('reading_file_example.txt')
line = f.readline()
print(type(line))
print(line)
f.close()

# readlines(): read all the text line by line and returns a list of lines
f = open('reading_file_example.txt')
lines = f.readlines()
print(type(lines))
print(lines)
f.close()

# Another way to get all the lines as a list is using splitlines():

f = open('reading_file_example.txt')
lines = f.read().splitlines()
print(type(lines))
print(lines)
f.close()

with open('reading_file_example.txt') as f:
    lines = f.read().splitlines()
    print(type(lines))
    print(lines)
    
with open('reading_file_example.txt','a') as f:
    f.write('This text has to be appended at the end')
    
with open('writing_file_example.txt','w') as f:
    f.write('This text will be written in a newly created file')


# import os
# if os.path.exists('example.txt'):
#     os.remove('example.txt')
# else:
#     os.remove('The file does not exist')
    
# dictionary
person_dct= {
    "name":"Sebastian",
    "country":"Argentina",
    "city":"Buenos Aires",
    "skills":["JavaScrip", "React","Python"]
}
# JSON: A string form a dictionary
person_json = "{'name': 'Sebastian', 'country': 'Argentina', 'city': 'Buenos Aires', 'skills': ['JavaScrip', 'React', 'Python']}"

# we use three quotes and make it multiple line to make it more readable
person_json = '''{
    "name":"Sebastian",
    "country":"Argentina",
    "city":"Buenos Aires",
    "skills":["JavaScrip", "React","Python"]
}'''
    
import json
# JSON
person_json = '''{
    "name":"Sebastian",
    "country":"Argentina",
    "city":"Buenos Aires",
    "skills":["JavaScrip", "React","Python"]
}'''
# let's change JSON to dictionary
person_dct = json.loads(person_json)
print(person_dct)
print(person_dct['name'])

import json
# python dictionary
person = {
    "name":"Sebastian",
    "country":"Argentina",
    "city":"Buenos Aires",
    "skills":["JavaScrip", "React","Python"]
}
# let's convert it to  json
person_json = json.dumps(person, indent=4) # indent could be 2, 4, 8. It beautifies the json
print(type(person_json))
print(person_json)

import json
# python dictionary
person = {
    "name":"Sebastian",
    "country":"Argentina",
    "city":"Buenos Aires",
    "skills":["JavaScrip", "React","Python"]
}
# let's convert 
with open('json_example.json', 'w', encoding='utf-8') as f:
    json.dump(person, f, ensure_ascii=False, indent=4)
    
import csv
with open('csv_example.csv') as f:
    csv_reader = csv.reader(f, delimiter=',') # w use, reader method to read csv
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are :{", ".join(row)}')
            line_count += 1
        else:
            print(
                f'\t{row[0]} is a teachers. He lives in {row[1]}, {row[2]}.')
            line_count += 1
    print(f'Number of lines:  {line_count}')

import xml.etree.ElementTree as ET
tree = ET.parse('xml_example.xml')
root = tree.getroot()
print('Root tag:', root.tag)
print('Attribute:', root.attrib)
for child in root:
    print('field: ', child.tag)
    

#exercise 1
f = open('obama_speech.txt')
lines = f.readlines()
print('Number of lines: ', len(lines))
lines = ''.join(lines)
print ("The number of words: " +  str(len(lines.split()))) 
f.close()

f = open('michelle_obama_speech.txt')
lines = f.readlines()
print('Number of lines: ', len(lines))
lines = ''.join(lines)
print ("The number of words: " +  str(len(lines.split()))) 
f.close()

f = open('donald_speech.txt')
lines = f.readlines()
print('Number of lines: ', len(lines))
lines = ''.join(lines)
print ("The number of words: " +  str(len(lines.split()))) 
f.close()

f = open('melina_trump_speech.txt')
lines = f.readlines()
print('Number of lines: ', len(lines))
lines = ''.join(lines)
print ("The number of words: " +  str(len(lines.split()))) 
f.close()


def most_spoken_language(file, number):
    import json 
  
    # JSON file 
    f = open (file, "r" , encoding='utf-8') 
  
    # Reading from file 
    data = json.loads(f.read()) 
  
    # Iterating through the json 
    # list 
    # print(data)
    data_dct = json.dumps(data, indent=4, sort_keys=True)
    import ast
    data_dct = ast.literal_eval(data_dct)
    print(data_dct)
    lst_aux = []
    # print(type(data_dct))
    for i in data_dct:
        lst_aux.append(i['languages'])
    print('before')
    print(lst_aux)
    lst_aux = [ number for row in lst_aux for number in row]
    print('after')
    print(lst_aux)
    from collections import Counter
    most_spoken_language = Counter(lst_aux).most_common(number)
    f.close()
    return most_spoken_language
    # Closing file 
 
print(most_spoken_language('./data/countries_data.json',10)) 

def most_populated_countries(file, number):
    import json 
    from collections import Counter
    # JSON file 
    f = open (file, "r" , encoding='utf-8') 
  
    # Reading from file 
    data = json.loads(f.read()) 
  
    # Iterating through the json 
    # list 
    # print(data)
    data_dct = json.dumps(data, indent=4, sort_keys=True)
    import ast
    data_dct = ast.literal_eval(data_dct)
    dct_aux = []
    aux = []
    aux1 = []
    # print(type(data_dct))
    for i in data_dct:
        aux.append(i['name'])
        aux1.append(i['population'])
        
    dct_aux = dict(zip(aux, aux1))
    
    k = Counter(dct_aux) 
    most_populated_countries = k.most_common(number)
    for i in most_populated_countries: 
        print(i[0]," :",i[1]," ")
    f.close()
    return most_populated_countries
    # Closing file 
 
print(most_populated_countries('./data/countries_data.json',3)) 

f = open('./data/email_exchanges_big.txt')
import re
txt = f.read()
print(type(txt))
regex_pattern = r'[\w\.-]+@[\w\.-]+'  # . any character, + any character one or more times 
match = re.findall(regex_pattern, txt)
print(match)
f.close()

def clean_txt(file):
  with open(file) as txt:
      mylist = [line.rstrip('\n') for line in txt]
      str1 = ""
      for ele in mylist:  
          str1 += ele
      sentence  = re.sub('[^A-Za-z ]', '', str1)
  return sentence #string

def find_most_common_words(file,number):    
    # Python program to find the k most frequent words 
    # from data set 
    from collections import Counter
    # split() returns list of all the words in the string 
    split_it = clean_txt(file).split()
    # Pass the split_it list to instance of Counter class. 
    Counter = Counter(split_it)
    # most_common() produces k frequently encountered 
    # input values and their respective counts. 
    most_common_words = Counter.most_common(number) 
    return most_common_words


print(find_most_common_words('./data/romeo_and_juliet.txt', 10))
    
print(find_most_common_words('obama_speech.txt', 3))
print(find_most_common_words('michelle_obama_speech.txt', 3))
print(find_most_common_words('donald_speech.txt', 3))
print(find_most_common_words('melina_trump_speech.txt', 3))

def remove_support_words(txt):
    with open('./data/stop_words.py', "r" ) as stop_words:
        lst_stop_words = stop_words.read()
        sentence = ''.join(lst_stop_words)
        regex_pattern = r'[A-Za-z]+'  # ^ in set character means negation, not A to Z, not a to z, no space
        matches = re.findall(regex_pattern, sentence)
        resultwords  = [word for word in txt if word.lower() not in matches]
        
        return resultwords

def most_frequent_words(file):    
    # Python program to find the k most frequent words 
    # from data set 
    from collections import Counter
    txt = clean_txt(file)
    regex_pattern = r'[A-Za-z]+'  # ^ in set character means negation, not A to Z, not a to z, no space
    matches = re.findall(regex_pattern, txt)
    txt_re = remove_support_words(matches)
    aux = Counter(txt_re).keys() # equals to list(set(words))
    aux1 = Counter(txt_re).values() # counts the elements' frequency
    aux, aux1 = zip(*sorted(zip(aux, aux1)))
    dct = dict(zip(aux, aux1))
    return dct

def compare_two_dct(dict1,dict2):
    dct = {}
    for key in dict1:
        if key in dict2:
            if key in dict2:
                
                if dict1[key] > dict2[key]:
                    dct[key] = int(dict2[key])
                else:
                    dct[key] = int(dict1[key])
    a = sorted(dct.items(), key=lambda x: x[1], reverse=True) 
    return a
            

def check_text_similarity (file1, file2):
    
    dct_txt1  = most_frequent_words(file1)
    dct_txt2  = most_frequent_words(file2)
    return compare_two_dct(dct_txt1,dct_txt2)

    
    
print(check_text_similarity('obama_speech.txt','donald_speech.txt'))

import csv
with open('./data/hacker_news.csv') as f:
    csv_reader = csv.reader(f, delimiter=',') # w use, reader method to read csv
    python    =0
    javascript=0
    java      =0
    id_person =[]
    for row in csv_reader:
            aux = ", ".join(row)
            if 'python' in aux or 'Python' in aux:
                python     +=1
            if 'JavaScript' in aux or 'Javascript' in aux or 'javascript' in aux:
                javascript +=1
            if 'Java' in aux or 'java' in aux:
                java       +=1

print('Number of lines containing Python: ',python)
print('Number of lines containing JavaScript: ',javascript)
print('Number of lines containing Java: ',java)