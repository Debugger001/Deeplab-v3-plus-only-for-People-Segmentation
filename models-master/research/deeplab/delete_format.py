train = open("trainval_list.txt", "r")
# train_list = tuple(train)
newtrain = open("new_trainval.txt", "w+")
# print(len(train_list))
for i in range(887):
    name = ""
    name = train.read(16)
    name = name.rstrip()
    name = name.split('.')[0]
    newtrain.write(name + '\n')
train.close()
newtrain.close()

# val = open("val.txt", "r")
# val_list = tuple(val)
# newval = open("new_val.txt", "w+")
# for i in range(len(val_list)):
#     name = ""
#     name = val.read(16)
#     name = name.rstrip()
#     name = name.split('.')[0]
#     newval.write(name + '\n')
# val.close()
# newval.close()
