import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.stats import linregress
import app.aux_func as af
import os

PATH = str(os.path.abspath(__file__))
PATH = PATH.replace(PATH.split("/")[-1], "")

PARAM = {'vert': 'a', 
         'hori': 'h',
         'multi': 'CI',
         'var': 'theta'}

G_DICT = {}
G_DICT['5'] = 'a=5'
G_DICT['11'] = 'a=11'
G_DICT['8'] = 'a=8' 

CI_DICT = {}
CI_DICT['WE'] = 'WE1'
CI_DICT['CB'] = 'CB'

LEG_DICT = {}
LEG_DICT['a'] = G_DICT
LEG_DICT['CI'] = CI_DICT

LINETYPE = ['--', '-.', ':']

def Sup_linear_fit(state=None, a=None, h=None, limit=None, break_n=3, lim_frac=1):
    print("-----------------------------------Start----------------------------------\n")
    # Create a directory to store the files
    os.makedirs(f'{PATH}fit_files/', exist_ok=True)
    path = f'{PATH}fit_files/'
    measures = af.file_crawler()['measures']
    print(f"Files: {len(measures)}\n")
    # Check if there is a "msd_files" folder in this directory in PATH
    if 'msd_files' not in os.listdir(PATH):
        print("Run MSD_plot first")
        return

    # Select one of the msd_files
    if a==None or h==None or state==None:
        msd_files = os.listdir(f'{PATH}msd_files')
        print("Arquivos:")
        for i in range(len(msd_files)):
            if ".txt" in msd_files[i]:
                print("\t"+str(i)+") "+ msd_files[i])
        msd_file = f'{PATH}msd_files/' + msd_files[int(input("Digite o índice do arquivo desejado: "))]
        # get h value from msd file
        h = msd_file.split("/")[-1].split("_")[4]
        a = msd_file.split("/")[-1].split("_")[2]
        state = msd_file.split("/")[-1].split("_")[5]
        state = state.replace(".txt", "")
    else:
        msd_file = f'{PATH}msd_files/msd_a_{a}_h_{h}_{state}.txt'

    plt.rcParams.update({'font.size': 13})
    fo_values = np.sort(np.unique(np.array([af.fo_from_file(f) for f in measures])))
    print("\nResultado:"+str(fo_values))
    print("\nOpções:")
    if limit == None:
        for i in range(len(fo_values)):
            print('\t'+fo_values[i])
        limit = input(r"Digite os valores de mu separados por espaço: ")
    # Transform the values into a list of floats
    limit = [i for i in limit.split()]
    limit = np.sort(np.array(limit))
    print("\nlimit: "+str(limit))
    limit_index = np.array([i for i in range(len(fo_values)) if fo_values[i] in limit]) +1
    print("limit_index: "+str(limit_index))

    # fit = input("Digite fit se deseja fazer a regressão linear por partes:") 
    data = np.loadtxt(msd_file, unpack=True)
    # filter the limit_index's in data
    dados = {}
    dados = {'dt': data[0,:]}
    dt = dados['dt']
    data = data[limit_index,:]

    # create a dictionary with the folowing indexes [t] + [fo_values]
    for i in range(len(limit)):
        dados[limit[i]] = data[i,:]

    for i in range(1, len(dados.items())):
        msd = np.array(list(dados.values())[i])
        # Create a logarithmically spaced list of indices
        indices = np.logspace(0, np.log10(len(dt)-1), 20).astype(int)
        # Define the color for the current m
        fo = str(list(dados.keys())[i])
        # Index of the neerest value to dt[-1]*lim_frac
        print(len(dt))
        lim_index = np.where(dt < dt[-1]*lim_frac)[0][-1]

        # Interactive fit
        print(f"\n\t Supervised fit of CI={state}, a={a}, h={h}, mu={fo}")
        dt_fit, slopes, breaks, intercepts = fitting(dt[1:lim_index], msd[1:lim_index])
        # Plot MSD over time with a label containing the value of a and the defined color
        for j in range(len(dt_fit)):
            msd_fit = np.exp(intercepts[j] + slopes[j]*np.log(dt_fit[j]))
            #plt.plot(dt_fit[j], np.exp(intercepts[j] + slopes[j]*np.log(dt_fit[j])), LINETYPE[j], label=f'slope={slopes[j]:.2f}', color=af.COLORS[i-1])
            plt.plot(dt_fit[j], msd_fit, LINETYPE[j], label=f'slope={slopes[j]:.2f}', color=af.COLORS[i-1])
        plt.plot(dt, msd, 'o', label=r'$\mu$'+f'={fo}', color=af.COLORS[i-1], markevery = indices)
        if len(breaks)<1:
            breaks = [np.log(dt[lim_index])]
        print(breaks)
        plt.axvline(x=np.exp(breaks[0]), color=af.COLORS[i-1], linestyle='-')
        with open(f"{path}fited_h_{h}_a_{a}_{state}.txt", 'a') as out:
            out.write(f"{fo}, {np.array(slopes)}, {np.array(breaks)}, {np.array(intercepts)}\n")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(1, dt[-1]*lim_frac)

    # Add labels and legend
    plt.title(f'CM MSD fit for {G_DICT[a]} and {state}')
    plt.xlabel(r'$\Delta$ t')
    plt.ylabel(r'MSD')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 5})
    plt.legend(loc='upper left', prop={'size': 7})
    # Create some space on the right side of the plot for the legend
    plt.subplots_adjust(right=0.15)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{PATH}fit_msd_a_{a}_h_{h}_{state}.png", dpi=300)
    # clear plt
    plt.clf()
    #return(slopes, breaks, intercepts)

def fitting(dt, msd):
    plt.plot(dt, msd, 'o')
    plt.yscale("log")
    plt.xscale("log")
    plt.title(f"How many regimes do you see?")
    plt.show()
    
    num_fits = int(input("How many fits do you want to do? "))
    
    interactive_fit = InteractiveFit(dt, msd, num_fits)
    input("Press Enter when done.")
    
    dt_fit, slopes, breaks, intercepts = interactive_fit.fit_and_plot()
    return(dt_fit, slopes, breaks, intercepts)

def fit_and_plot(dt, msd, intervals):
    Lines = {"dt_fit": [], "msd_fit": [],"slopes": [], "intercepts": []}
    plt.figure()
    for i, (start, end) in enumerate(intervals):
        mask = (dt >= start) & (dt <= end)
        slope, intercept, _, _, _ = linregress(np.log(dt[mask]), np.log(msd[mask]))
        dt_fit = np.linspace(start, end, num=100)
        msd_fit = np.exp(intercept + slope * np.log(dt_fit))
        Lines["slopes"].append(slope)
        Lines["intercepts"].append(intercept)
        Lines["dt_fit"].append(dt_fit)
        #Lines["msd_fit"].append(msd_fit)
        plt.plot(dt_fit, msd_fit, '--', label=f'slope={slope:.2f}', color=f'C{i}')
        plt.plot(dt[mask], msd[mask], 'o', color=f'C{i}')
        print(f"Interval {i+1}: slope={slope}, intercept={intercept}")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.show()

    # Calculate breaks
    breaks = []
    for i in range(len(Lines["slopes"]) - 1):
        m1, b1 = Lines["slopes"][i], Lines["intercepts"][i]
        m2, b2 = Lines["slopes"][i + 1], Lines["intercepts"][i + 1]
        x_break = (b2 - b1) / (m1 - m2)
        breaks.append(x_break)
        plt.axvline(x=np.exp(x_break), color=f'C{i}', linestyle='--')
        print(f"Break {i}: x={x_break}")

    return Lines["dt_fit"], Lines["slopes"], breaks, Lines["intercepts"]

# Interactive selection
class InteractiveFit:
    def __init__(self, dt, msd, num_fits):
        self.dt = dt
        self.msd = msd
        self.num_fits = num_fits
        self.intervals = []
        self.fig, self.ax = plt.subplots()
        self.ax.plot(dt, msd, 'o')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.title.set_text("Select the intervals using your cursor.")
        self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=True)
        plt.show()

    def onselect(self, xmin, xmax):
        self.intervals.append((xmin, xmax))
        print(f"Selected interval: {xmin} to {xmax}")
        if len(self.intervals) >= self.num_fits:
            plt.close(self.fig)

    def fit_and_plot(self):
        return fit_and_plot(self.dt, self.msd, self.intervals)

def MSD_diagram(var = PARAM['var'], config=None, limit='all', break_n=3, lim_frac=0.1):
    print("-----------------------------------Start----------------------------------\n")
    measures = af.file_crawler()['measures']
    fo_values = np.sort(np.unique(np.array([float(af.fo_from_file(f)) for f in measures])))
    print("\nResultado:"+str(fo_values))
    print("\nOpções:")
    if limit == None:
        for i in range(len(fo_values)):
            print('\t'+fo_values[i])
        limit = input(r"Digite os valores de mu separados por espaço: ")
    if limit == 'all':
        limit = ''
        for i in fo_values:
            if i != 0:
                if i<1:
                    limit += f'{i:.1f} '
                else:
                    limit += str(int(i))+' '
            else:
                limit += '0 '
    # Transform the values into a list of floats
    limit = [i for i in limit.split()]
    limit = np.sort(np.array(limit))
    print("\nlimit: "+str(limit))
    limit_index = np.array([i for i in range(len(fo_values)) if fo_values[i] in limit]) +1
    print("limit_index: "+str(limit_index))
    
    # Check if there is a "msd_files" folder in this directory in PATH
    if 'msd_files' not in os.listdir(PATH):
        print("\nERRO!!!! Run MSD_plot first")
        return
    # --------------------------------------------------------------------------------------
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

    breaks = []
    slopes = []
    dr = []
    var_dict = {'p_length': dr, 'p_time': breaks, 'slope': slopes}

    if var == 'p_length':
        y_limits = (0,20000)
        labeling_y = r'$\Delta r_{p}$'
        leg_loc = "upper left"
    elif var == 'p_time':
        y_limits = (0, 3000)
        labeling_y = r"$\Delta t_{p}$"
        leg_loc = "upper left"
    elif var == 'slope':
        y_limits = (0, 2)
        labeling_y = r"$\alpha_1$"
        leg_loc = "lower right"
    else:
        print("Invalid 'var' parameter")
        return

    #list of all files in {PATH}msd_files/
    msd_files = os.listdir(f'{PATH}msd_files')
    msd_files = [f for f in msd_files if '.txt' in f]
    search_dict = {'a': -4, 'h': -2, 'CI': -1}
    a_values = np.sort(np.unique(np.array([float(f.split("/")[-1].split("_")[search_dict['a']]) for f in msd_files])))
    # remove values 6, 7, 9 and 10
    a_values = a_values[(a_values != 6) & (a_values != 7) & (a_values != 9) & (a_values != 10)]
    h_values = np.sort(np.unique(np.array([float(f.split("/")[-1].split("_")[search_dict['h']]) for f in msd_files])))
    CI_values = ['CB', 'WE']
    print(msd_files)

    config_dict = {'a': a_values, 'h': h_values, 'fo': fo_values, 'CI': np.array(CI_values)}
    vertical = config_dict[PARAM['vert']]
    horizontal = config_dict[PARAM['hori']]
    curves = config_dict[PARAM['multi']]

    #--------------------------------------------------------------------------------------
    # Create a subplot
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(len(vertical), len(horizontal), figsize=(int(len(horizontal)*5), int(len(vertical)*4)))
    legend_added = {}

    for msd_file in msd_files:
        vert_index = msd_file.split("/")[-1].split("_")[search_dict[PARAM['vert']]].replace(".txt", "")
        try:
            vert_index = float(vert_index)
        except:
            vert_index = vert_index
        hori_index = msd_file.split("/")[-1].split("_")[search_dict[PARAM['hori']]].replace(".txt", "")
        try:
            hori_index = float(hori_index)
        except:
            hori_index = hori_index
        curve_index = msd_file.split("/")[-1].split("_")[search_dict[PARAM['multi']]].replace(".txt", "")
        try:
            curve_index = float(curve_index)
        except:
            curve_index = curve_index

        # Find the corresponding subplot
        if vert_index in vertical:
            Row = vertical.tolist().index(vert_index)
        else:
            continue
        if hori_index in horizontal:
            Col = horizontal.tolist().index(hori_index)
        else:
            continue
        if curve_index in curves:
            cur = curves.tolist().index(curve_index)
        else:
            continue

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
        if ax not in legend_added:
            legend_added[ax] = []
            if var in ['p_length', 'p_time']:
                ax.set_yscale("log")
                #ax.set_xscale("log")
            #ax.set_yscale("log")
            #ax.set_xscale("log")
            ax.set_ylim(y_limits)
            ax.set_xlim(fo_values[1], fo_values[-3]+0.5)
            # Add labels and legend
            ax.set_title(f'{PARAM["vert"]}={vert_index} and {PARAM["hori"]}={hori_index}')
            ax.set_xlabel(r'$\mu$')
            ax.set_ylabel(labeling_y)
            # Set xticks
            ax.set_xticks(fo_values.astype(int)[:-2])

        # fit = input("Digite fit se deseja fazer a regressão linear por partes:") 
        data = np.loadtxt(PATH+'msd_files/'+msd_file, unpack=True)
        # filter the limit_index's in data
        dados = {}
        dados = {'dt': data[0,:]}
        dt = dados['dt']
        data = data[1:,:]
        
        # create a dictionary with the folowing indexes [t] + [fo_values]
        for i in range(len(limit)):
            dados[limit[i]] = data[i,:]

        var_dict = {}
        new_fit_data = False
        if 'fit_files' not in os.listdir(PATH):
            new_fit_data = True
        for i in range(1, len(dados.items())):
            msd = np.array(list(dados.values())[i])
            # Create a logarithmically spaced list of indices
            state = msd_file.split("/")[-1].split("_")[-1].replace(".txt", "")
            fo = str(list(dados.keys())[i])
            a = msd_file.split("/")[-1].split("_")[-4]
            h = msd_file.split("/")[-1].split("_")[-2]

            if new_fit_data:
                print(f"\nERRO!!! Run MSD_sup_fit to generate fit data.")
                return ()
                #slopes, breaks, intercepts = Sup_linear_fit(state, a, h, fo)
            else:
                path = PATH + 'fit_files/'
                files = os.listdir(path)
                files = [f for f in files if f.endswith('.txt')]
                file = [f for f in files if f'a_{a}' in f and f'h_{h}' in f and f'{state}' in f]
                aux = False
                if len(file) > 0:
                    aux = True
                    for line in open(path+"/"+file[0]):
                        #print("\n Linha",line)
                        #print("\n Path",file[0])
                        mu, slopes_str, breaks_str, intercepts_str = line.split(",")
                        if float(mu) == float(fo):
                            slopes = np.fromstring(slopes_str.replace('[','').replace(']',''), sep=' ')
                            breaks = np.fromstring(breaks_str.replace('[','').replace(']',''), sep=' ')
                            intercepts = np.fromstring(intercepts_str.replace('[',']').replace(']',''), sep=' ')
                            break

            if aux:
                # Find the coresponding msd point for the break[0] point
                # print(f"fo={fo}, slopes={slopes}, breaks={breaks}")
                break_index = np.where(dt < np.exp(breaks[0]))[0][-1]
                dr = msd[break_index]
                var_dict['p_length'] = np.sqrt(dr)
                var_dict['p_time'] = np.exp(breaks[0])
                var_dict['slope'] = slopes[0]

                if f"{af.COLORS[cur]}" not in legend_added[ax]:
                    ax.plot(float(fo), var_dict[var], af.MARKERS[cur], label=f"{PARAM['multi']}={LEG_DICT[PARAM['multi']][curve_index]}", color=af.COLORS[cur])
                    legend_added[ax].append(f"{af.COLORS[cur]}")
                else:
                    ax.plot(float(fo), var_dict[var], af.MARKERS[cur], color=af.COLORS[cur])

    for ax in legend_added:
        ax.legend(loc=leg_loc, prop={'size': 10})
    
    # Create some space on the right side of the plot for the legend
    plt.subplots_adjust(right=0.15)
    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{PATH}fit_{var}_diagram_{PARAM['vert']}-vs-{PARAM['hori']}.png", dpi=300)
    # clear plt
    plt.clf()

if __name__ == "__main__":
    MSD_diagram('p_length', ('h', 'a'))
    #MSD_diagram('p_time', ('h', 'a'))
    MSD_diagram('slope', ('h', 'a'))
    #Sup_linear_fit('WE', '11', '10', '0 0.1 0.5 1 2 3 4 5 6 7 8 9 10')
    #Sup_linear_fit('CB', '11', '10', '0 0.1 0.5 1 2 3 4 5 6 7 8 9 10')
    #Sup_linear_fit('WE', '8', '10', '0 0.1 0.5 1 2 3 4 5 6 7 8 9 10')
    #Sup_linear_fit('CB', '8', '10', '0 0.1 0.5 1 2 3 4 5 6 7 8 9 10')
    #Sup_linear_fit('WE', '5', '10', '0 0.1 0.5 1 2 3 4 5 6 7 8 9 10')
    #Sup_linear_fit('CB', '5', '10', '0 0.1 0.5 1 2 3 4 5 6 7 8 9 10')