'''python strings'''
x = 'This is a string'
y = "This is a string too"
z = "let's go"
#escape charcter turns slash into string
w = 'let\'s go'

#multiplying a string by x produces x times the string
x1 = x*2 
#adding strings- concatenates thems
x2 = x + y
#to check for data type use type() method
type(x)
r = 1.345
type(r)
#rounding to nearest integer
round(r)

'''User input'''
#note, user inputs are strings so to turn them to inetegers we use int() method
p = int(input('Enter integer'))
print(p)
type(p)

#end script execution use quit()

#logic
'''Use if, elif and else statements to implement logic'''

#Iterations
'''for loop'''
s = [1,2,3]
for i in s:  #i takes a different value of the list for each iteration
    print(i)
    
#range function
t = range(10) # from 0 t0 10, ten not inclusive
t = range(2,10) # from 2 t0 10, ten not inclusive
t = range(2,10,2) # from 2 t0 10, ten not inclusive step size of 2
for i in t:  
    print(i)
    
'''while loop''' # combines for loop and if statement
#while(condition):

'''Functions'''
#def Function_name(arguments):
#   code

#default parameter
# def function_name(arguments =x)

'''Data structures'''
    '''1. Lists'''
x = [1,2,3,4,5,6] #sams data type
#indexing
print(x[0]) #first element is 1
print(x[1:4]) #print elements from index 1 to 3, 4 exclusive
print(x[:4]) # from index zero to 3
print(x[3:]) #from index 3 to the end
print(x[-2]) #second item from the end

for i, y in enumerate(x): #prints index in i and vale in y
    print(i, y)