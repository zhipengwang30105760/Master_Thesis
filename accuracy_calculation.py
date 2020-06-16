target_list =[[[522, 4], [4, 1]], [[522, 4], [4, 1]], [[522, 4], [4, 1]], [[522, 4], [4, 1]], [[349, 2], [2, 1]], [[349, 2], [3, 0]], [[349, 2], [2, 1]], [[349, 2], [3, 0]]]
result = []
#print(len(target_list))
#print(type(target_list[0][0][0]))
for a in target_list:
    #print(type(a[0][0]))
    score = (a[0][0] + a[1][1]) / (a[0][0] + a[1][1] + a[0][1] + a[1][0])
    result.append(score)

print(result)