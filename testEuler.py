


list = xrange(1000)



max_list = []
max_prod = 0
for i in list[:len(list)-14]:

	helper = list[i:i+14]
	product = 1

	for j in helper:
		product = product*j
	if product > max_prod:
		max_prod = product
		max_list = helper

print max_list
