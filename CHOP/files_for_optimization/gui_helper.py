from IPython.display import display
import ipywidgets as widgets
import gui_helper_opt
import pickle
import os


diff_min = -30
diff_max = 30
diff_range = [diff_min/2, diff_max/2]
ratio_min = -1000
ratio_max = 1000
ratio_range = [ratio_min/2, ratio_max/2]

keys = ["DV1/2 Act", "GV Slope (%WT)", "DV1/2 SSI", "SSI Slope (%WT)", "Tau Fast (%WT)", "Tau Slow (%WT)",
       "Tau 0mV (%WT)", "Persistent (%WT)", "Peak Amp (%WT)"]
non_percentage_indices = [0, 2]
ratio_widgets = dict()
diff_widgets = dict()

layout = widgets.Layout(width='60%', height='40px')
style = {'description_width': 'initial'}

for i in range(len(keys)):
    if i in non_percentage_indices:  
        w = widgets.FloatSlider(
            value=0,
            min=diff_min,
            max=diff_max,
            step=0.01,
            description=keys[i],
            readout=True,
            layout = layout,
            style = style
        )
        diff_widgets[keys[i]] = w
    else:
        w = widgets.FloatSlider(
            value=0,
            min=ratio_min,
            max=ratio_max,
            step=0.01,
            description=keys[i],
            readout=True,
            layout = layout,
            style = style
        )
        ratio_widgets[keys[i]] = w

        
other_widgets = []
other_widgets.append(widgets.IntSlider(
    value=5,
    min=0,
    max=3000,
    step=1,
    description='# of offsprings:',
    layout = layout,
    style = style
))

other_widgets.append(widgets.IntSlider(
    value=10,
    min=0,
    max=3000,
    step=1,
    description='# of generations:',
    layout = layout,
    style = style
))

other_widgets.append(widgets.Text(
    value='na12',
    placeholder='Type something',
    description='channel name:',
    layout = layout,
    style = style,
    disabled=False
))


other_widgets.append(widgets.Text(
    value='neoWT',
    placeholder='Type something',
    description='mutant name:',
    layout = layout,
    style = style,
    disabled=False
))

button = widgets.Button(
    description='Run optimization',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='check'
)

other_widgets.append(button)

other_widgets.append(widgets.Checkbox(
    value=False,
    description='use the Test Queue',
    disabled=False,
    indent=False
))



def run_opt(b):
    config_dict = dict()
    for w in ratio_widgets.values():
        try:
            config_dict[w.description] = w.value
        except:
            continue
    
    for w in other_widgets:
        try:
            config_dict[w.description] = w.value
        except:
            continue
        
    for w in diff_widgets.values():
        try:
            config_dict[w.description] = w.value
        except:
            continue
    
    with open('opt_config.pickle', 'wb') as handle:
        pickle.dump(config_dict, handle)
        
    import subprocess

    bashCommand = "sbatch gui_job_debug.slr"
    # os.system(bashCommand)
    print(os.popen(bashCommand).read())
    
    

button.on_click(run_opt)




def diaplay_all():
    print("Here are the percentage parameters (from -1000 to 1000):\n")
    for w in ratio_widgets.values():
        display(w)
        
    print("\n\n\nHere are the difference parameters (from -30 to 30):\n")
    for w in diff_widgets.values():
        display(w)
        
    print("\n\n\nPlease specify number of generations and offsprings (from 0 to 3000):\n")
    display(other_widgets[0])
    display(other_widgets[1])
    
    print("\n\n\nPlease specify channel and mutant names:\n")
    display(other_widgets[2])
    display(other_widgets[3])
    
    
    print("\n")
    display(other_widgets[5])
    
    print("\n")
    display(other_widgets[4])
    