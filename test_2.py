
a = range(10)
b= range(10,20)
c = ['a', 'b', 'c'] 


print zip(a,b,c)
print zip(*zip(a,b,c))