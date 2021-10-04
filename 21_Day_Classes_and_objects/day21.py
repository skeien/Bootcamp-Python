# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:04:35 2020

@author: sebas
"""

class Person:
  pass
p = Person()
print(p)

class Person:
      def __init__ (self, name):
          self.name =name

p = Person('Sebastian')
print(p.name)
print(p)

class Person:
      def __init__(self, firstname, lastname, age, country, city):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city


p = Person('Sebastian', 'Keienburg', 230, 'Argentina', 'Buenos Aires')
print(p.firstname)
print(p.lastname)
print(p.age)
print(p.country)
print(p.city)

class Person:
      def __init__(self, firstname, lastname, age, country, city):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city

      def person_info(self):
        return f'{self.firstname} {self.lastname} is {self.age} years old. He lives in {self.city}, {self.country}'


p = Person('Sebastian', 'Keienburg', 230, 'Argentina', 'Buenos Aires')
print(p.person_info())

class Person:
      def __init__(self, firstname='Sebastian', lastname='Keienburg', age=230, country='Argentina', city='Buenos Aires'):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city

      def person_info(self):
        return f'{self.firstname} {self.lastname} is {self.age} years old. He lives in {self.city}, {self.country}.'

p1 = Person()
print(p1.person_info())
p2 = Person('John', 'Doe', 30, 'Nomanland', 'Noman city')
print(p2.person_info())

class Person:
      def __init__(self, firstname='Sebastian', lastname='Keienburg', age=230, country='Argentina', city='Buenos Aires'):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city
          self.skills = []

      def person_info(self):
        return f'{self.firstname} {self.lastname} is {self.age} years old. He lives in {self.city}, {self.country}.'
      def add_skill(self, skill):
          self.skills.append(skill)

p1 = Person()
print(p1.person_info())
p1.add_skill('HTML')
p1.add_skill('C')
p1.add_skill('JavaScript')
p2 = Person('John', 'Doe', 30, 'Nomanland', 'Noman city')
print(p2.person_info())
print(p1.skills)
print(p2.skills)

class Student(Person):
    pass


s1 = Student('Eyob', 'Yetayeh', 30, 'Finland', 'Helsinki')
s2 = Student('Lidiya', 'Teklemariam', 28, 'Finland', 'Espoo')
print(s1.person_info())
s1.add_skill('JavaScript')
s1.add_skill('React')
s1.add_skill('Python')
print(s1.skills)

print(s2.person_info())
s2.add_skill('Organizing')
s2.add_skill('Marketing')
s2.add_skill('Digital Marketing')
print(s2.skills)

class Student(Person):
    def __init__ (self, firstname='Sebastian', lastname='Keienburg', age=230, country='Argentina', city='Buenos Aires', gender='male'):
        self.gender = gender
        super().__init__(firstname, lastname,age, country, city)
    def person_info(self):
        gender = 'He' if self.gender =='male' else 'She'
        return f'{self.firstname} {self.lastname} is {self.age} years old. {gender} lives in {self.city}, {self.country}.'

s1 = Student('Eyob', 'Yetayeh', 30, 'Finland', 'Helsinki','male')
s2 = Student('Lidiya', 'Teklemariam', 28, 'Finland', 'Espoo', 'female')
print(s1.person_info())
s1.add_skill('JavaScript')
s1.add_skill('React')
s1.add_skill('Python')
print(s1.skills)

print(s2.person_info())
s2.add_skill('Organizing')
s2.add_skill('Marketing')
s2.add_skill('Digital Marketing')
print(s2.skills)

#1
ages = [31, 26, 34, 37, 27, 26, 32, 32, 26, 27, 27, 24, 32, 33, 27, 25, 26, 38, 37, 31, 34, 24, 33, 29, 26]
class data:
    def count():
        return len(ages)
    def suma():
        return sum(ages)
    def mini():
        return min(ages)
    def maxi():
        return max(ages)
    def ranger():
        return max(ages)-min(ages)
    def mean():
        return sum(ages)/len(ages)
    def median():
        lst = ages
        n = len(lst)
        s = sorted(lst)
        return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None
    def mode():
        counter = 0
        num = ages[0]  
        for i in ages: 
            curr_frequency = ages.count(i) 
            if(curr_frequency> counter): 
                counter = curr_frequency 
                num = i 
        return "%d "%num +"count: "+"%.2f"%counter
    def std():
        mean = sum(ages) / len(ages)   # mean
        var  = sum(pow(x-mean,2) for x in ages) / len(ages)  # variance
        return (var)**(1/2)  # standard deviation
    def var():
        mean = sum(ages) / len(ages)   # mean
        return sum(pow(x-mean,2) for x in ages) / len(ages)  # variance
    def freq_dist():
        d = {}
        for i in ages:
            if d.get(i):
                d[i] += 1
            else:
                 d[i] = 1
        return d

print('Count:', data.count()) # 25
print('Sum: ', data.suma()) # 744
print('Min: ', data.mini()) # 24
print('Max: ', data.maxi()) # 38
print('Range: ', data.ranger()) # 14
print('Mean: ', data.mean()) # 30
print('Median: ', data.median()) # 29
print('Mode: ', data.mode()) # {'mode': 26, 'count': 5}
print('Standard Deviation: ', data.std()) # 4.2
print('Variance: ', data.var()) # 17.5
print('Frequency Distribution: ', data.freq_dist()) # [(20.0, 26), (16.0, 27), (12.0, 32), (8.0, 37), (8.0, 34), (8.0, 33), (8.0, 31), (8.0, 24), (4.0, 38), (4.0, 29), (4.0, 25)]

class PersonAccount:
      def __init__(self, firstname='Sebastian', lastname='Keienburg'):
          self.firstname          = firstname
          self.lastname           = lastname
          self.incomes            = []
          self.expenses_properties = []

      def total_income(self):
        return sum(self.incomes)
      def total_expense(self):
        return sum(self.expenses_properties)
      def account_info(self):
          print('name: ',self.firstname)
          print('lastname: ',self.lastname)
          total_incomes = sum(self.incomes)
          print('Total Income:  ',total_incomes)
          total_expenses = sum(self.expenses_properties)
          print('Total expense:  ',total_expenses)
          net_worth = total_incomes - total_expenses 
          return f'{self.firstname} {self.lastname} has net worth: {net_worth}'
      def add_income(self,income):
        self.incomes.append(income)
      def add_expense(self,expense):
        self.expenses_properties.append(expense)
      def account_balance(self):
          total_incomes = sum(self.incomes)
          total_expenses = sum(self.expenses_properties)
          net_worth = total_incomes - total_expenses 
          return f'{self.firstname} {self.lastname} has net worth: {net_worth}'
        
p1 = PersonAccount()     
p1.add_income(6000)
p1.add_income(4000)
p1.add_expense(500)
p1.add_expense(500)
print('Total Income:', p1.total_income())
print(p1.account_info())