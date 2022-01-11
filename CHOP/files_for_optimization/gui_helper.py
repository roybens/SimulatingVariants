from IPython.display import display
import ipywidgets as widgets
import gui_helper_opt

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

for i in range(len(keys)):
    if i in non_percentage_indices:  
        w = widgets.FloatSlider(
            value=0,
            min=diff_min,
            max=diff_max,
            step=0.01,
            description=keys[i],
            readout=True
        )
        diff_widgets[keys[i]] = w
    else:
        w = widgets.FloatSlider(
            value=0,
            min=ratio_min,
            max=ratio_max,
            step=0.01,
            description=keys[i],
            readout=True
        )
        ratio_widgets[keys[i]] = w

        
other_widgets = []
other_widgets.append(widgets.IntSlider(
    value=5,
    min=0,
    max=3000,
    step=1,
    description='# of offsprings:'
))

other_widgets.append(widgets.IntSlider(
    value=10,
    min=0,
    max=3000,
    step=1,
    description='# of generations:'
))

other_widgets.append(widgets.Text(
    value='na12',
    placeholder='Type something',
    description='channel name:',
    disabled=False
))


other_widgets.append(widgets.Text(
    value='neoWT',
    placeholder='Type something',
    description='mutant name:',
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
    gui_helper_opt.run_opt(other_widgets, ratio_widgets, diff_widgets)
    
    

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
    