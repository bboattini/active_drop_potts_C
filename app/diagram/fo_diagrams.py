import matplotlib.pyplot as plt
import numpy as np
import os
import app.aux_func as af
#from app.MSD.MSD_plot import G_DICT

PATH = str(os.path.abspath(__file__))
PARAM = {'vert': 'a', 
         'hori': 'h',
         'multi': 'CI',
         'var': 'theta'}
IC_DICT = {'CB': 'CB', 'WE': 'WE1'}

def Diagram_Generator(var = PARAM['var'], config=None):
    print("-----------------------------------Start----------------------------------\n")
    #--------------------------------------------------------------------------------------
    # User input
    if config == None:
        print(
'''Insert the variables(CI, a or h) to be used in vertical and horizontal axis, respectively
(separate by a space):''')
        # Convert the input into a tuple
        config = tuple(input().split())
    PARAM['vert'] = config[0]
    PARAM['hori'] = config[1]
    PARAM['multi'] = [l for l in ['a', 'h', 'CI'] if l not in [PARAM['vert'], PARAM['hori']]][0]
    
    if var == 'theta':
        var_dir = "peak_detect"
        y_limits = (90, 160)
        labeling_y = r'$\theta_c$(°)'
        leg_loc = "lower right"
    elif var == 'Vp':
        var_dir = "histograms"
        y_limits = (0, 0.2)
        labeling_y = r"$f$"
        leg_loc = "upper right"
    elif var == 'e':
        var_dir = "peak_detect"
        y_limits = (0, 1)
        labeling_y = r"$e$"
        leg_loc = "lower right"
    else:
        print("Invalid 'var' parameter")
        return
    
    #--------------------------------------------------------------------------------------
    # Histograms file setup
    path = PATH.replace(PATH.split("/")[-1], "").replace("/"+PATH.split("/")[-2], "")
    path = path + var_dir
    # Find the files inside path that ends in .txt
    files = os.listdir(path)
    files_to_read = [f for f in files if f.endswith('.txt') and f.startswith(var)]
    # remove the value 10 and 6 from a_values
    files_to_read = [f for f in files_to_read if 'a_6' not in f and 'a_10' not in f]
    '''files_to_read = [f for f in files_to_read if 'a_7' not in f and 'a_8' not in f]
    files_to_read = [f for f in files_to_read if 'a_9' not in f]'''

    if len(files_to_read) > 0:
        print(f"Found {len(files_to_read)} files")
    else:
        print("No files found, please run the peak detection algorithm first.")
        return()

    # Sort the parameters list from lower to greater
    h_values = np.sort(np.unique(np.array([float(h_from_file(f)) for f in files_to_read])))
    a_values = np.sort(np.unique(np.array([float(a_from_file(f)) for f in files_to_read])))
    
    # remove de value 8 from a_values
    a_values = np.delete(a_values, np.where(a_values == 6))
    #a_values = np.delete(a_values, np.where(a_values == 7))
    #a_values = np.delete(a_values, np.where(a_values == 9))
    a_values = np.delete(a_values, np.where(a_values == 10))
    print(a_values)
    fo_values = np.sort(np.unique(np.array([0 for f in files_to_read])))

    config_dict = {'a': a_values, 'h': h_values, 'fo': fo_values, 'CI': np.array(["CB", "WE"])}
    vertical = config_dict[PARAM['vert']]
    horizontal = config_dict[PARAM['hori']]
    curves = config_dict[PARAM['multi']]

    #--------------------------------------------------------------------------------------
    # Create a subplot
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(len(vertical), len(horizontal), figsize=(int(len(horizontal)*5), int(len(vertical)*4)))
    legend_added = {}
    #--------------------------------------------------------------------------------------
    # File loop
    files = os.listdir(PATH.replace("fo_diagrams.py", ""))
    files = [f for f in files if f.endswith('.txt')]
    files  = [f for f in files if f.endswith('.txt') and not f.startswith('a6') and not f.startswith('a10')]
    #files  = [f for f in files if f.endswith('.txt') and not f.startswith('a7') and not f.startswith('a9')]
    #files  = [f for f in files if f.endswith('.txt') and not f.startswith('a8')]
    print(files)
    flag = 0
    for f in files:
        if 'a' in [PARAM['vert'], PARAM['hori']]:
            if 'a' == PARAM['vert']:
                vert_index = float(f.split("_")[-2].replace("a", ""))
                hori_index = 10
            else:
                hori_index = float(f.split("_")[-2].replace("a", ""))
                vert_index = 10
        elif 'CI' in [PARAM['vert'], PARAM['hori']]:
            if 'CI' == PARAM['vert']:
                vert_index = vertical[flag]
                hori_index = 10
            else:
                vert_index = 10
                hori_index = horizontal[flag]
            flag += 1
        # Find the corresponding subplot
        Row = vertical.tolist().index(vert_index)
        Col = horizontal.tolist().index(hori_index)

        if len(vertical) == 1 and len(horizontal) == 1:
            ax = axs
        elif isinstance(axs, np.ndarray):
            if axs.ndim > 1:
                ax = axs[Row, Col]
            else:
                if len(vertical) == 1: # or axs[Col/Row], depending on which dimension is 1
                    ax = axs[Col]
                else:
                    ax = axs[Row]

        # Get the set of states that have been added to the legend for this axis, or create a new set if it doesn't exist yet
        #legend_added_for_this_ax = legend_added.setdefault(ax, set())
        legend_added[ax] = []

        if flag == 0 and var == 'theta':
            # Load the minima file
            minima = np.loadtxt(PATH.replace("fo_diagrams.py", "")+f, unpack=True)
            if minima.ndim == 0:
                minima = [minima.item()]  # Convert to a list with a single element
            x = np.arange(0, 12)
            for m in minima:
                ax.plot(x, m*np.ones(len(x)), color='black', alpha=0.5, linewidth=1)
        if flag == 2:
            break

    for f in files_to_read:
        if 'CI' not in [PARAM['vert'], PARAM['hori']]:
            vert_index = float(index_finder(PARAM['vert'], f))
            hori_index = float(index_finder(PARAM['hori'], f))
            '''if hori_index != 8:
                ax = find_ax(axs, vert_index, hori_index, vertical, horizontal)'''
        if hori_index == 6 or hori_index == 10:
            continue 
        print(f"for {PARAM['hori']} = {hori_index} and {PARAM['vert']} = {vert_index}")

        for line in open(path+"/"+f):
            state, fo, means_str, stds_str = line.split(",")
            if 'CI' in [PARAM['vert'], PARAM['hori']]:
                if 'CI' == PARAM['vert']:
                    vert_index = state
                    hori_index = float(index_finder(PARAM['hori'], f))
                else:
                    vert_index = float(index_finder(PARAM['vert'], f))
                    hori_index = state
                multi_index = float(index_finder(PARAM['multi'], f))
            else:
                multi_index = state
                vert_index = float(index_finder(PARAM['vert'], f))
                hori_index = float(index_finder(PARAM['hori'], f))
            ax = find_ax(axs, vert_index, hori_index, vertical, horizontal)
            means = np.fromstring(means_str.replace("[","").replace("]",""), sep=' ')
            stds = np.fromstring(stds_str.replace("[","").replace("]",""), sep=' ')
            fo = float(fo)

            for color_idx, c in enumerate(curves):
                if multi_index == c:
                    # Order means vector in descending order
                    means = means[np.argsort(means)[::-1]]
                    # Restrict the number of means to 1
                    means = [means[0]]
                    # if len(means) > 2:
                    #    means = [means[0], (means[1]+means[2])/2]
                    for idx, m in enumerate(means):                    
                        if f"{af.COLORS[color_idx]}{idx}" not in legend_added[ax]:
                            #ax.errorbar(fo, m, yerr=np.abs(stds[idx]), fmt=af.MARKERS[color_idx], color=af.COLORS[color_idx], ecolor=af.COLORS[color_idx], elinewidth=1, capsize=0, label=f'{c} '+r'$\theta$'+f'{idx+1}', alpha=1/(idx+1))
                            ax.errorbar(fo, m, yerr=np.abs(stds[idx]), fmt=af.MARKERS[color_idx], color=af.COLORS[color_idx], ecolor=af.COLORS[color_idx], elinewidth=1, capsize=0, label='IC = '+IC_DICT[c], alpha=1/(idx+1))
                            # markerfacecolor='none' to make the marker transparent
                            #legend_added_for_this_ax.add(f'{af.COLORS[color_idx]}{idx}')
                            legend_added[ax].append(f'{af.COLORS[color_idx]}{idx}')
                            selected = np.where(np.array(list(legend_added[ax])) == f'{af.COLORS[color_idx]}{idx}')[0][0]
                            ax.plot(fo, m, color=af.COLORS[selected], linewidth=5)
                            #ax.errorbar(fo, m, yerr=np.abs(stds[idx]), fmt=".", color=af.COLORS[selected], ecolor=af.COLORS[selected], elinewidth=1, capsize=0, label=f"{c} "+r'$\theta_c$')
                            #ax.set_title(f"{PARAM['hori']} = {hori_index} and {PARAM['vert']} = {vert_index}")
                            ax.set_title(f"{PARAM['hori']} = {hori_index}")
                            #ax.set_title(f"{G_DICT[str(int(hori_index))]}")
                            ax.set_ylabel(labeling_y)
                            ax.set_xlabel(r'$\mu$')
                            ax.set_xlim(0, 10.5)
                            ax.set_ylim(y_limits)
                            # location down right
                            ax.legend(loc=leg_loc, fontsize='small')
                        else:
                            ax.errorbar(fo, m, yerr=np.abs(stds[idx]), fmt=af.MARKERS[color_idx], color=af.COLORS[color_idx], ecolor=af.COLORS[color_idx], elinewidth=1, capsize=0, alpha=1/(idx+1))
                            # markerfacecolor='none' to make the marker transparent
                            selected = np.where(np.array(list(legend_added[ax])) == f'{af.COLORS[color_idx]}{idx}')[0][0]
                            ax.plot(fo, m, color=af.COLORS[selected], linewidth=5)
                            #ax.errorbar(fo, m, yerr=np.abs(stds[idx]), fmt=".", color=af.COLORS[selected], ecolor=af.COLORS[selected], elinewidth=1, capsize=0)
        
            
        # add space between suplots
        plt.tight_layout()
        #print(f"\nState: {state}\nfo: {fo}\nmeans: {means}\nstds: {stds}")
    #--------------------------------------------------------------------------------------
    # Save the figure
    save_dir = str(os.path.abspath(__file__)) 
    save_dir = save_dir.replace('/'+save_dir.split('/')[-1], "")
    fig.tight_layout()
    fig.savefig(f"{save_dir}/{var}_mu_diagram_{PARAM['vert']}-vs-{PARAM['hori']}.jpg", format='jpeg')
    print("-----------------------------------End------------------------------------\n")
    return()

def index_finder(tag, file):
    if tag == 'a':
        return(a_from_file(file))
    elif tag == 'h':
        return(h_from_file(file))
    elif tag == 'fo':
        return(fo_from_file(file))
    elif tag == 'CI':
        return(state_from_file(file))

def a_from_file (File):
  fs = File.split("/")[-1].split("_")
  fl = len(fs)
  pad = int(fs[-1].replace(".txt", ""))
  return pad # return a value

def h_from_file (File):
  fs = File.split("/")[-1].split("_")
  fl = len(fs)
  pad = int(fs[-3].replace("", ""))
  return pad # return h value

def state_from_file (File):
  fs = File.split("/")[-1].split("_")
  fl = len(fs)
  #pad = fs[0].replace("/", "")
  #return pad # return state value

def fo_from_file (File):
  fs = File.split("/")[-1].split("_")
  fl = len(fs)
  #pad = fs[-1]
  #return pad # return fo value

def find_ax(axs, vert_index, hori_index, vertical, horizontal):
    # Find the corresponding subplot
    Row = vertical.tolist().index(vert_index)
    Col = horizontal.tolist().index(hori_index)

    if len(vertical) == 1 and len(horizontal) == 1:
        ax = axs
    elif isinstance(axs, np.ndarray):
        if axs.ndim > 1:
            ax = axs[Row, Col]
        else:
            if len(vertical) == 1: # or axs[Col/Row], depending on which dimension is 1
                ax = axs[Col]
            else:
                ax = axs[Row]  
    return ax

if __name__ == '__main__':
    Diagram_Generator(var='Vp', config = ('h', 'a'))
    #Diagram_Generator(var='Vp', config = ('h', 'CI'))
    Diagram_Generator(var='theta', config = ('h', 'a'))
    #Diagram_Generator(var='theta', config = ('h', 'CI'))
    #Diagram_Generator(var='e', config = ('h', 'a'))
    #Diagram_Generator(var='e', config = ('h', 'CI'))
