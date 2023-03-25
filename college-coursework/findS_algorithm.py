
#taking input data 
instance=int(input("enter how many instances in training data: "))
d=[]
h=[]
s=[]
for i in range(instance):
    ip=input("input instance:").split()
    d.append(ip)


# initializing most specific hypothesis
for i in range(len(d[0])-1):
    h.append("phi")
    s.append("phi")


# number of columns with feature values
j=len(d[0])-1; 



    
# Find S algorithm

for i in range(instance):
     #if positive response update h
    if d[i][j]=="yes":       
        if h==s:
            # for first positive response
            h=d[i]
            h.pop(j) #to remove last element in the data which is the response / output
            
        else:
            
            # check element by element and update
            for k in range(j):
                                
                # if the element consistent skip
                if d[i][k]==h[k]:
                    continue
               
                # else generalize the individual element
                else:
                    h[k]="?"
                
    
    #if negative response ignore
    else:
        continue
                

print("\n\nhypothesis: ",h)        

test = input("\n\ninput test case :")
for t in range(j-1):
    if h[t]== test[t] or h[t]=="?":
        r="yes"
    else:
        r="no"
        break
print("response: ",r)
