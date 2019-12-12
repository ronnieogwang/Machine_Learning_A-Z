import matplotlib.pyplot as plt
# we use this for plotting graphs

x = [1,2,3,4]
y = [13,15,17,19]

x2 = [1,2,3,4]
y2 = [20,24,28,32]

plt.plot(x,y, label='first line')
plt.plot(x2,y2,label='Second line')

#labels
plt.xlabel('x number')
plt.ylabel('y number')

#title
plt.title('Y vs X')

#legend
plt.legend()

plt.show()

#bar graph
plt.bar(x,y,color='c')
plt.show

#histogram
plt.hist(x,y,histtyp='bar',rwidth=0.8)# width creates spaces btn the bars, they do not take up all space

#scatter plots. show correlation
plt.scatter(x,y, color='k')
