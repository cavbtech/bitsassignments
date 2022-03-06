def test():
    #opent the file in read mode
    file = open("inputPS1.txt","r")
    stats = 0

    #iterate over the file line by line and take the input
    for line in file :
        #first line to ignore the testcase
        if stats == 0:
            stats += 1;
            continue
        #take the deadline and store it in the list
        elif stats%2 == 1 :
            deadlines = list(map(int,line.split()))
        #take the bonus and store them in the list
        elif stats %2 == 0:
            bonus = list(map(int,line.split()))
            #create an array which store the bonus that we will get each day
            day = [0]* (max(deadlines)+1)

            #combine bonus and deadlines list and then sort them in non-increasing order
            question = list(zip(bonus,deadlines))
            question.sort(reverse = True)

            #now for each of the question just check if the slot is available or not
            for profit, endline in question:
                i = endline
                #check for the slot
                while i > 0:
                    if day[i] == 0:
                        #then assign it the value equal to bonus
                        day[i] = profit
                        break
                    else :
                        i -= 1
            #now print the total profit in the output file
            file = open("outputPS5.txt","a")
            file.write(str(sum(day))+"\n")


        stats += 1

if __name__ == '__main__':
    test()