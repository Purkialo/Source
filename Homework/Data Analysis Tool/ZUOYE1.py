my_str= input("Input str: ")
print(my_str)
lenth = len(my_str)
total_length = 1
length = 1
center = 0
flag = 0
for i in range(len(my_str)):
    length = 1
    try:
        if(my_str[i] == my_str[i + 1]):
            length = 2
            flag = 1
            while(i - flag >= 0 and i + flag + 1 < len(my_str)):
                if(my_str[i - flag] == my_str[i + flag + 1]):
                    length =length + 2
                flag = flag + 1
            if(length > total_length):
                total_length = length
                center = i
    except Exception:
        continue
    try:
        if(my_str[i - 1] == my_str[i + 1]):
            length = 3
            flag = 2
            while(i - flag >= 0 and i + flag < len(my_str)):
                if(my_str[i - flag] == my_str[i + flag]):
                    length =length + 2
                flag = flag + 1
            if(length > total_length):
                total_length = length
                center = i
    except Exception:
        continue
print(center)
print(total_length)
