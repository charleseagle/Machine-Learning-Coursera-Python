from __future__ import division
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np


def DisplayData(X, example_width = None):
    if example_width == None:
        example_width = int(np.round(np.sqrt(np.shape(X)[1])))
    
    m, n = np.shape(X)
    example_height = int(n/example_width)
    
    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m/display_rows))
    
    # Beteen images padding
    pad = 1
    
    # Setup blank display
    display_array = -np.ones((pad+display_rows*(example_height + pad), \
                pad + display_cols * (example_width + pad)))
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            # Copy the patch and get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex,:]))
            initial_x = pad + j * (example_height + pad)
            initial_y =pad + i * (example_width + pad)
            display_array[initial_x:initial_x+example_height, \
		             initial_y:initial_y+example_width] = \
                      X[curr_ex, :].reshape(example_height, example_width)\
                      / max_val
            curr_ex += 1
        if curr_ex > m:
            break


    # Display image
    img = scipy.misc.toimage(display_array.T)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)  
    plt.show()  
        