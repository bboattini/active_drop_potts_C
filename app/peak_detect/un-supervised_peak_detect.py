import os
import app.aux_func as af
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator

PATH = str(os.path.abspath(__file__)).replace("peak_detect/un-supervised_peak_detect.py","")
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '#FF5733', '#DAF7A6']
PARAM = {
    'var': 'B',
    'a': 0,
    'h': 0,
    'w': 5,
    'min_theta': 6,
    'samp_frac': 10,
    'bin': 5
}

def Un_Sup_Peak_Det(var=None, a=None, h=None):
    print("-----------------------------------Start----------------------------------\n")
    files_to_read = af.file_crawler()['measures']
    fo_values = np.sort(np.unique(np.array([af.fo_from_file(f) for f in files_to_read])))
    a_values = np.sort(np.unique(np.array([af.a_from_file(f) for f in files_to_read])))
    h_values = np.sort(np.unique(np.array([af.h_from_file(f) for f in files_to_read])))
    #--------------------------------------------------------------------------------------
    # User input
    if var == None:
        print("Choose between 'B' or 'theta': ")
        PARAM['var'] = input()
    else:
        PARAM['var'] = var

    if a == None:
        print("Chose a value for 'a': ")
        for a in a_values:
            print(f'\t{a:.2f}')
        PARAM['a'] = input()
    else:
        PARAM['a'] = a

    if h == None:
        print("Chose a value for 'h': ")
        for h in h_values:
            print(f'\t{h}')
        PARAM['h'] = input()
    else:
        PARAM['h'] = h 
    #--------------------------------------------------------------------------------------
    # Selecting the files to read
    h = PARAM['h']
    a = PARAM['a']
    files_to_read = [f for f in files_to_read if f'h_{h}' in f]
    files_to_read = [f for f in files_to_read if f'a_{a}' in f]

    columns = len(fo_values)
    #--------------------------------------------------------------------------------------
    # Figure init
    fig = plt.figure(figsize=(7*columns,10))
    fig.suptitle('a = '+ str(a) + ' and h = ' + str(h), fontsize=25)
    plt.rcParams['font.size'] = '20'

    #--------------------------------------------------------------------------------------
    # Graphics generator loop
    for f in files_to_read:
        
        t, V, Vw, E, bxw, byw, rxw, ryw, txw, tyw, vbw, nulo1, nulo2, x_CM, y_CM, z_CM = np.loadtxt(f, unpack=True)
        state = af.state_from_file(f)
        fo = af.fo_from_file(f)
        print(fo)

        if PARAM['var'] == 'theta':
            xTS = txw
            yTS = tyw
            x_limits = (80, 160)
            labeling_x = r"$\theta_c$"
            labeling_y = r"$Freq(\theta_c)$"
            cri = PARAM['min_theta']
        elif PARAM['var'] == 'B':
            xTS = bxw
            yTS = byw
            x_limits = (15, 70)
            labeling_x = r"$B_x$"
            labeling_y = r"$B_y$"
            cri = PARAM['w']
        elif PARAM['var'] == 'e':
            folders = [fol for fol in os.listdir(PATH+'video/') if fol.startswith('movie')]
            folders = [fol for fol in folders if f'movie{state}' in fol]
            folders = [fol for fol in folders if f'_fo{fo}' in fol]
            x_limits = (0, 1)
            labeling_x = r"$e$"
            if len(folders) < 1:
                # Skip this for loop iteration
                continue
            else:
                t, x0, y0, a1, a2, e = np.loadtxt(f"{PATH}video/{folders[0]}/ellipses.txt", unpack=True)

            xTS = e
            yTS = []
        else:
            print("Invalid 'var' parameter")
            return

        N = 1
        leg_loc = 'lower left'
        if state == 'WE':
            N = columns+1 
            leg_loc = 'upper left'
        
        # Find the list index of fo in sorted float fo_values

        fo_loc = np.where(np.sort(np.array(fo_values, dtype=float)) == float(fo))[0][0]
        print(f"\n~~~~~~~~Plotting histogram a = {PARAM['a']}, f = {fo} and {state}~~~~~~~~")

        #================================HISTOGRAMS================================
        sample = len(t)//PARAM['samp_frac']
        ax = plt.subplot(2,columns,N+fo_loc)

        xTS = xTS[sample:]
        if PARAM['var'] != 'e':
            yTS = yTS[sample:]
            train_data = np.append(bxw[sample:], byw[sample:]).reshape(-1, 1)
        else:
            train_data = xTS.reshape(-1, 1)
        
        best_k = best_k_estimator(train_data, 1, 10)
        k = -1
        pred = best_k.predict(train_data)

        if PARAM['var'] != 'e':
            valid_data = np.append(xTS, yTS).reshape(-1, 1)
        else:
            valid_data = xTS.reshape(-1, 1)
        
        while k < 1:

            means = np.array([np.mean(valid_data[pred==c]) for c in np.unique(pred)])
            classes = np.unique(pred)
            freq_TS = [classes[0]]*len(classes)

            for c in range(len(classes)):
                freq_TS[c] = af.time_series_to_frequency_array(valid_data[pred == classes[c]], bin=PARAM['bin'])

            maximum = np.max([np.max(freq_TS[c][:,1]) for c in range(len(classes))])
            for c in range(len(classes)):
                if k == 0:
                    plt.bar(freq_TS[c][:,0], freq_TS[c][:,1]/maximum, width = 1/PARAM['bin'], color = af.COLORS[c], align='center', label=f"m={means[c]:.2f}")
                    plt.plot(means[c]*np.ones(2), [1.3,0], color='k')

                    plt.yscale("log")
                    plt.xlim(x_limits)
                    plt.ylim(10**(-5), 1.3)
                    plt.legend(loc=leg_loc)
                    plt.title(labeling_x + r' $\mu$='+ str(fo) +' state ' + state)

            max_freq = np.ones(len(classes))
            for c in range(len(classes)):
                max_freq[c] = np.max(freq_TS[c][:,1])
            aux = -1
            
            while len(np.unique(pred)) != aux:
                aux = len(np.unique(pred))
                if PARAM['var'] == 'e':
                    PARAM['w'] = 0.01
                pred = merge_classes(pred, means, max_freq, criterion=PARAM['w'])
                classes = np.unique(pred)
                # Updating means and max_freqs
                means = np.array([np.mean(valid_data[pred==c]) for c in classes])
                stds = np.array([np.std(valid_data[pred==c]) for c in classes])
                max_freq = np.array([np.max(freq_TS[c][:,1]) for c in range(len(classes))]) 
            # order the means acoording to max_freq.sort()
            means = means[np.argsort(max_freq)]
            stds = stds[np.argsort(max_freq)]

            #plt.show()
            k += 1
        # Final adjustments
        save_dir = str(os.path.abspath(__file__)) 
        save_dir = save_dir.replace('/'+save_dir.split('/')[-1], "")
        ax.set_xlabel(labeling_x)
        ax.set_ylabel(labeling_y)
        fig.tight_layout()
        fig.savefig(f"{save_dir}/{PARAM['var']}_hist_h_{PARAM['h']}_a{PARAM['a']}.pdf", format='pdf')
        # Add a line in a txt file with the respective columns: fo [m for m in means]
        with open(f"{save_dir}/{PARAM['var']}_means_h_{PARAM['h']}_a_{PARAM['a']}.txt", 'a') as out:
            out.write(f"{state}, {fo}, {means}, {stds}\n")
    print("-----------------------------------End------------------------------------\n")

def merge_classes(pred, means, freqs, criterion):
  #classes = np.array(list(dict.fromkeys(pred)))
  classes = np.unique(pred)
  # Create a mapping from pred classes to means
  class_to_mean = {pred_class: mean for pred_class, mean in zip(classes, means)}
  class_to_freq = {pred_class: freq for pred_class, freq in zip(classes, freqs)}
  aux_class_to_mean = class_to_mean.copy()
  aux_class_to_freq = class_to_freq.copy()
  aux = False
  # Iterate over each class and its corresponding mean
  for c1 in range(len(classes)):
    current_mean = class_to_mean[classes[c1]]
    current_freq = class_to_freq[classes[c1]]
    # Find classes that should be merged based on the condition
    for c2 in range(c1, len(classes)):
      other_mean = class_to_mean[classes[c2]]
      if len(means) > 1:
        min_diff = np.sort(np.abs(means - current_mean))[1]
      else:
        continue
      if abs(current_mean - other_mean) == min_diff and min_diff < criterion:
        '''
        other_freq = class_to_freq[classes[c2]]
        # Witch is the higer value of the two freq
        if current_freq > other_freq:
            frac = other_freq/current_freq
        else:
            frac = current_freq/other_freq
        if frac > p:
        '''
        if True:
            print(f"Encontrei um par de classes para merge: {classes[c1]} e {classes[c2]}")
            print(class_to_freq)
            # Merge 'other_class' into 'current_class' by updating 'pred'
            pred[pred == classes[c2]] = classes[c1]
            # Remove 'other_class' from 'class_to_mean' to prevent further comparisons
            if classes[c2] in list(aux_class_to_mean.keys()):
                del aux_class_to_mean[classes[c2]]
                del aux_class_to_freq[classes[c2]] 
                aux = True
                break
    if aux:
        break

  return(pred)

def best_k_estimator(data, init, fin):
# Step 1: Initialize list for WCSS
  wcss = []
  k_range = range(init,fin)

  for k in k_range:
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    wcss.append(kmeans.inertia_)
  knee_locator = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
  best_k = KMeans(n_clusters=knee_locator.knee, random_state=0).fit(data)
  #best_k = KMeans(n_clusters=3, random_state=0).fit(data)
  return(best_k)


if __name__ == '__main__':
    Un_Sup_Peak_Det(var='theta', a=5, h=10)
    Un_Sup_Peak_Det(var='theta', a=6, h=10)
    Un_Sup_Peak_Det(var='theta', a=7, h=10)
    Un_Sup_Peak_Det(var='theta', a=8, h=10)
    Un_Sup_Peak_Det(var='theta', a=9, h=10)
    Un_Sup_Peak_Det(var='theta', a=10, h=10)
    Un_Sup_Peak_Det(var='theta', a=11, h=10)
    #Un_Sup_Peak_Det(var='e', a=5, h=10)
    #Un_Sup_Peak_Det(var='e', a=11, h=10)
    #Un_Sup_Peak_Det(var='e', a=8, h=10)