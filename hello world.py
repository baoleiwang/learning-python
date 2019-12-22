from pip._vendor.distlib.compat import raw_input

number  = raw_input("please input three int",)
number = int(number)
a = number%10
b = (number%100-a) / 10
c = (number-a-b*10)/100
count = a * 100 +b * 10 +c
print(count)


