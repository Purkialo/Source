my_str= input("Input str: ")
print(my_str)
total_center = 0
center = 0
total_length = 1
length = 1
for i in range(len(my_str)):
    center = i
    length = 1
    if((i - length) < 0 or (i+length) >= len(my_str)):
        continue
    else:
        if(int(center - length) < 0 or int(center + length) >= len(my_str)):
            break
        while(my_str[int(center - length)] == my_str[int(center + length)]):
            length = length + 1
            if(length > total_length):
                total_length = length
                total_center = center
            if(int(center - length) < 0 or int(center + length) >= len(my_str)):
                break
        center = i + 0.5
        length = 0.5
        if(int(center - length) < 0 or int(center + length) >= len(my_str)):
            break
        while(my_str[int(center - length)] == my_str[int(center + length)]):
            length = length + 1
            if(length > total_length):
                total_length = length
                total_center = center
            if(int(center - length) < 0 or int(center + length) >= len(my_str)):
                break
if(total_length % 1 != 0):
    total_length = int(2 * (total_length - 0.5))
else:
    total_length = int(2 * (total_length - 1) + 1)
print(total_length)