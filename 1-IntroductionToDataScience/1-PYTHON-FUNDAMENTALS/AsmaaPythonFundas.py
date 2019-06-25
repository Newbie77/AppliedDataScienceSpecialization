# author: Asmaa ~ 2019
# Coursera Introduction to data science course
# WEEK1 Python Fundamentals

# Functions
# basic function


def add_str(a, b, c):
    return a + ' ' + b+' '+c


print(add_str('hello', 'world', '!'))

# function overloading concept


def add_strs(a, b, c=None):
    if(c == None):
        return(a+' '+b)
    else:
        return(a+' '+b+' '+c)


print(add_strs('hello', 'world', '!'), add_strs('hi', 'world'))

# assigning func to a variable
str_fun = add_strs
print(str_fun('hello', 'world', '!'))

# type operator
print(type('asmaa'), type(str_fun))

# data structures examples
my_tuple = (1, 2, 3, 1.15, 'asmaa') # immutable
my_list = [5, 4 , 9, 44.15, 'asmaa'] # mutable
my_list.append(16)

# foreach
for val in my_list:
    print(val)

# list operations
my_list = my_list + [77, 88]
my_list = my_list * 2
print(my_list)

print(1 in my_list)

# slicing
sub_list = my_list[1:3] 
sub_list = my_list[:5]
sub_list = my_list[2:]


# strings
my_str = 'Hello, my name is asmaa'
str_list = my_str.split(' ')
print(str_list)

# dictionaries
# declare
my_dict = {'name': 'asmaa', 'surname': 'mirkhan' }

# modify
my_dict['name'] = 'esma'
print(my_dict)

# iterate
for item in my_dict:
    print(item, my_dict[item])

# unpacking
sample_list = ['asmaa', 'mirkhan']
name, surname = sample_list
print(name, surname)

# string formatting
my_str='Hello {}, my name is {}'
my_str = my_str.format('world', 'asmaa')
print(my_str)

# csv operations