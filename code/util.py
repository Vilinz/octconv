import numpy as np
import matplotlib.pyplot as plt


def show_curve_double(ys0, ys1, title, position='upper right'):
    """
    plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: loss or accuracy
    """
    x = np.array(range(len(ys0)))
    y0 = np.array(ys0)
    y1 = np.array(ys1)
    plt.plot(x, y0, c='b', label='Resnet')
    plt.plot(x, y1, c='r', label='OCtaveResNet')
    plt.legend(loc=position)
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.show()
    
    
def show_curve_double1(ys0, ys1, title, position='upper right'):
    """
    plot curlve for Loss and Accuacy
    Args:
        ys: loss or acc list
        title: loss or accuracy
    """
    x = np.array(range(len(ys0)))
    y0 = np.array(ys0)
    y1 = np.array(ys1)
    plt.plot(x, y0, c='b', label='OCtaveResNet')
    plt.plot(x, y1, c='r', label='HTOL')
    plt.legend(loc=position)
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.show()
    
    
def save_loss(loss, acc, filename_loss, filename_acc):
    f = open(filename_loss, 'w')
    f.write(loss)
    f.close()
    f = open(filename_acc, 'w')
    f.write(acc)
    f.close()
    
def get50_from_file(filename):
    result = []
    f = open(filename)
    data = f.read()
    data = data[1:-1].split(',')
    for i in range(50):
        result.append(float(data[i]))
    print(result)
    return result