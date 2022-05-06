arrow_tip = [295, 179]
center = [293, 269]

difference_of_x = abs(arrow_tip[0] - center[0])
difference_of_y = abs(arrow_tip[1] - center[1])


print(difference_of_x)
print(difference_of_y)

if difference_of_x < 6 and difference_of_y > 50 :
    # Up and Down
    if arrow_tip[1] < center[1]:
        print("up")
    else:
        print("down")


    pass

else:
    print("left ot right ")