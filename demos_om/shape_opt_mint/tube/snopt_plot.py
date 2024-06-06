import numpy as np
import matplotlib.pyplot as plt

f = open("./SNOPT_report/SNOPT_summary9.out")
lines = f.readlines()
major_inds = []
feasible_vals = []
opt_vals = []
merit_vals = []


major_ind = 0
feasible_ind = 4
opt_ind = 5
merit_ind = 6

record_flag = 0

for i in range(1, len(lines)):
    items_temp = list(filter(lambda x: x!='', lines[i].split(' ')))
    # print(items_temp)
    items = []
    for i0, item in enumerate(items_temp):
        if ')(' in item:
            item_split = item.split(')(')
            items += [item_split[0][1:], item_split[1][:-1]]
        else:
            items += [item]
    if items[0] == 'Major':
        record_flag += 1
        continue
    elif items[0] == '\n':
        next_items = list(filter(lambda x: x!='', lines[i+1].split(' ')))
        if next_items[0] == 'SNOPTC':
            break
        if next_items[0] == 'Minor':
            continue
    if items[0] == 'Minor':
        continue

    if record_flag > 0 and record_flag < 11:
        # record data
        ind = int(items[major_ind])
        if ind > 0 and ind != major_inds[-1]:
            major_inds += [int(items[major_ind]),]
            if '(' in items[feasible_ind]:
                feasible_vals += [float(items[feasible_ind][1:-1]),]
            else:
                feasible_vals += [float(items[feasible_ind]),]
            if '(' in items[opt_ind]:
                opt_vals += [float(items[opt_ind][1:-1]),]
            else:
                opt_vals += [float(items[opt_ind]),]
            if '(' in items[merit_ind]:
                merit_vals += [float(items[merit_ind][1:-1]),]
            else:
                merit_vals += [float(items[merit_ind]),]
            record_flag += 1
        elif ind == 0:
            major_inds += [int(items[major_ind]),]
            if '(' in items[feasible_ind-1]:
                feasible_vals += [float(items[feasible_ind-1][1:-1]),]
            else:
                feasible_vals += [float(items[feasible_ind-1]),]
            if '(' in items[opt_ind-1]:
                opt_vals += [float(items[opt_ind-1][1:-1]),]
            else:
                opt_vals += [float(items[opt_ind-1]),]
            if '(' in items[merit_ind-1]:
                merit_vals += [float(items[merit_ind-1][1:-1]),]
            else:
                merit_vals += [float(items[merit_ind-1]),]
            record_flag += 1
    else:
        record_flag = 0



fig, axs = plt.subplots(3, 1, sharex=True)
# Plot merit function
axs[0].plot(major_inds, merit_vals)
axs[0].set_ylabel('Merit function')
axs[0].set_yscale('log')
axs[0].grid()
axs[0].spines['top'].set_color('gray')
axs[0].spines['bottom'].set_color('gray')
axs[0].spines['left'].set_color('gray')
axs[0].spines['right'].set_color('gray')
axs[0].spines['top'].set_linewidth(0.5)
axs[0].spines['bottom'].set_linewidth(0.5)
axs[0].spines['left'].set_linewidth(0.5)
axs[0].spines['right'].set_linewidth(0.5)
axs[0].xaxis.set_ticks_position('none') 
axs[0].yaxis.set_ticks_position('none')

# Plot optimiality
axs[1].plot(major_inds, opt_vals)
axs[1].set_yscale('log')
axs[1].set_ylabel('Optimiality')
axs[1].grid()
# axs[1].spines['top'].set_color('none')
# axs[1].spines['bottom'].set_color('none')
# axs[1].spines['left'].set_color('none')
# axs[1].spines['right'].set_color('none')
axs[1].spines['top'].set_color('gray')
axs[1].spines['bottom'].set_color('gray')
axs[1].spines['left'].set_color('gray')
axs[1].spines['right'].set_color('gray')
axs[1].spines['top'].set_linewidth(0.5)
axs[1].spines['bottom'].set_linewidth(0.5)
axs[1].spines['left'].set_linewidth(0.5)
axs[1].spines['right'].set_linewidth(0.5)
axs[1].xaxis.set_ticks_position('none') 
axs[1].yaxis.set_ticks_position('none') 

# Plot feasiblity
axs[2].plot(major_inds, feasible_vals)
axs[2].set_yscale('log')
axs[2].set_ylabel('Feasibility')
axs[2].set_xlabel('Number of iterations')
axs[2].grid()
# axs[2].spines['top'].set_color('none')
# axs[2].spines['bottom'].set_color('none')
# axs[2].spines['left'].set_color('none')
# axs[2].spines['right'].set_color('none')
axs[2].spines['top'].set_color('gray')
axs[2].spines['bottom'].set_color('gray')
axs[2].spines['left'].set_color('gray')
axs[2].spines['right'].set_color('gray')
axs[2].spines['top'].set_linewidth(0.5)
axs[2].spines['bottom'].set_linewidth(0.5)
axs[2].spines['left'].set_linewidth(0.5)
axs[2].spines['right'].set_linewidth(0.5)
axs[2].xaxis.set_ticks_position('none') 
axs[2].yaxis.set_ticks_position('none')

plt.show()