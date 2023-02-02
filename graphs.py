import matplotlib.pylab as plt

def plot_graph(title, epoc, result):
    plt.figure()
    plt.plot(epoc, result, label= "Accuracy/RMSE")
    
    plt.legend(loc= 'lower left')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("epoch")
    plt.ylabel("accuracy/rmse")
    plt.savefig('result/' + title + '.png')
    plt.show()
    #plt.close()
    