# Plotting specific imports:
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display
import numpy as np



##  Plotting ##
def plot(x_name,y_names,  x_array, y_arrays,y_axis, title='Figure',
                legendlocation="best", y_colors=None,
                y_format =None, x_format =None,figsize=(6,4),grid=True,formatter=False): 
    
    '''
    Makes lineplots of the arrays using matplotlib

    Args:
            x_name(string)  : Name of x-axis
            y_names (list)  : Containing strings with the names of the lineplots
            x_array(array)  : Data for x-variable
            y_arrays(list)  : Containing arrays with data for the y-variable of all plots
            y_axis(string)  : Name of y_axis
            title(string)   : Figure title
            legendlocation(string): location of legend
            formatter(list) : List of formattting and tick location options
            figsize (tuple) : Figure size

    
    Returns:
            fig, ax         : matplotlib figure objects
    '''
    
    fig, ax = plt.subplots(figsize=figsize)


    if not y_colors:
        y_colors = [None for i in range(len(y_names))]
        jupyter lab
    for y_array,y_name,y_color in zip(y_arrays,y_names,y_colors):
        ax.plot(x_array,y_array,label=y_name,color=y_color)
    

    ax.legend(loc=legendlocation)



    #settings:
    ax.set_title(title)
    ax.grid(grid)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_axis)

    if  formatter:
        ax.xaxis.set_major_locator(formatter[0])
        ax.xaxis.set_major_formatter(formatter[1])

        ax.yaxis.set_major_locator(formatter[2])
        ax.yaxis.set_major_formatter(formatter[3])
        



    return fig, ax
'''    
def plot_hist(hist, edges, names=[''],tools="pan,wheel_zoom,box_zoom,reset,save",
             plot_range = False,x_label='x',y_label='y',title='Figure',
             alpha=0.5,legendlocation='top_right',width=500,height=500,
             fill_colors=['blue'],line_colors=['purple']):
    '''
    Plots a histogram using bokeh, the data is most easily be prepared using np.histogram()
    before being inputted into this function. 
    
    Args:
            hist  (list)        : Containing arrays with distribution of the data
            edges (list)        : Containing array with x-axis bins-location-data
            names (list)        : With names for the histograms if muliple are plotted
            tools (string)      : Bokeh tools
            plot_range(list)    : If you wish to decide the range of the x-axis, 
                this argument can be called as a list with: [min,max]
            x_label(string)     : Label of the x-axis
            y_label(string)     : Label of the y-axis
            title(string)       : Title of the figure
            fill_colors(list)   : Color(s) to fill the histogram(s), hex-color-code is also accepted
            line_colors(list)   : Color(s) in the line surrounding the histogram(s)
    
    Returns:
            p (bokeh.plotting.figure.Figure): The figure, which has to be called 
                in the bokeh.plotting comand, show(), to be viewed
    

    '''
    p = figure(title=title, tools=tools, x_axis_label=x_label, 
               y_axis_label=y_label,plot_width=width, plot_height=height)
    
    
    for h,e,name,fill_color,line_color in zip(hist,edges,names,fill_colors,line_colors):
        p.quad(top=h, bottom=0, left=e[:-1], right=e[1:],
               fill_color=fill_color, line_color=line_color, alpha=alpha,
              legend=name)
    
    p.y_range.start=0
    if plot_range == False:
        p.x_range.start = edges[0][0]
        p.x_range.end = edges[0][-1]
    else:
        p.x_range.start = plot_range[0]
        p.x_range.end = plot_range[-1]
    
    if names != ['']:
        p.legend.location = legendlocation
    
    return p
'''



##  2. AS-AD model ##

# We redifine our lambdifyed funtion to accept the par-dictionairy, and check that they give the same result:
def redefine_fun(lambda_fun):
    '''
    Calculates the equilibrium outputgab using the analytical solution
    derived in sympy.
    
    Args:
        lambda_fun (fun)    : Lambda function to be redifined    
    
    Returns:
        fun (fun)           : Redifened function

                This functions has arguements: 
                y_t1 (float)        : The outputgab in the previous period
                pi_t1 (float)       : The inflationgab in the previous period
                v_t (float)         : The demand chock in the current period
                s_t (float)         : The supply chock in the current period
                s_t1 (float)        : The supply chock in the previous period
                par (dict)          : Dictionairy contaning values of parameters

    '''
    
    fun = lambda y_t1,pi_t1,v_t,s_t,s_t1, par : lambda_fun(y_t1,pi_t1,v_t,s_t,s_t1,par['alpha'],par['gamma'],par['h'],par['b'],par['phi'])

    return fun

def ad(y_t,v_t,par):
    '''
    Aggregate demand
    
    Args:
        y_t (float)   : Outputgab in current period
        v_t (float)   : Demand chock in current period
        par (dict)    : Dictionairy contaning values of parameters
    Returns 
        ad (float)    : Aggregate demand
    
    '''
    h = par['h']
    alpha =par['alpha']
    b = par['b']
    
    ad = 1/(h*alpha)*(v_t-(1+b*alpha)*y_t)
    return ad

def sras(y_t, y_t1,pi_t1,s_t,s_t1,par):
    '''
    Short run aggregate supply
    
    Args:
        y_t (float)   : Outputgab in current period
        y_t1 (float)  : The outputgab in the previous period
        pi_t1 (float) : The inflationgab in the previous period
        s_t (float)   : Supply chock in current period
        s_t1 (float)  : Supply chock in previous period
        par (dict)    : Dictionairy contaning values of parameters
    Returns 
        sras (float)  : Short run aggregate supply    
    '''
    
    phi = par['phi']
    gamma = par['gamma']
    sras = pi_t1+gamma*y_t-phi*gamma*y_t1+s_t-phi*s_t1
    return sras

def d_pers(v_t1,x_t,par):

    '''
    Args:
        v_t1 (float)  : Demand chock in previous period
        x_t (float)   : Added demand chock in current period
    
    Returns:
        v_t (float)   : Demand chock in current period
    '''
    v_t = par['delta']*v_t1+x_t
    return v_t

def s_pers(s_t1,c_t,par):
    '''
    Args:
        s_t1 (float)  : Supply chock in previous period
        c_t (float)   : Added supply chock in current period
    
    Returns:
        s_t (float)   : Supply chock in current period
    '''
    s_t = par['omega']*s_t1+c_t
    return s_t





## 3. Exchange economy ##
'''
def utility(x1,x2,x3,beta1,beta2,beta3, gamma):
    utility = (x1**beta1*x2**beta2*x3**beta3)**gamma
    
    return utility


def utility_distribution(x1s,x2s,x3s,x1s_equal,x2s_equal,x3s_equal,betas,gamma,plot_range=[0,4]):

    '''
    Calculates the distribution of utility for all comsumer, for a given gamma and for two levels of comsumption for all comsumers,
    one derived from randomly distributed endowments, and one for equally distributed endowments
    Calculates the mean and variance, and makes a two figures containing everything

    Args:
            x1s (array)        : Comsumption of good 1 for each comsumer
            x2s (array)        : Comsumption of good 2 for each comsumer
            x3s (array)        : Comsumption of good 3 for each comsumer
            x1s_equal (array)  : Comsumption of good 1 for each comsumer (Equal distribution of endowments)
            x2s_equal (array)  : Comsumption of good 2 for each comsumer (Equal distribution of endowments)
            x2s_equal (array)  : Comsumption of good 3 for each comsumer (Equal distribution of endowments)
            betas (array)      : Containing beta for all comsumers for all goods
            gamma (float)      : Parameter
            plot_range(list)   : Containing min and max of range of the plotted x-axis. 
    Returns:
            plot1 (bokeh.plotting.figure.Figure) : The figure, for random endowments, which has to be called 
                in the bokeh.plotting comand, show(), to be viewed
            plot2 (bokeh.plotting.figure.Figure) : The figure, for equal endowments, which has to be called 
                in the bokeh.plotting comand, show(), to be viewed
    '''

    # Random endowments
    utilitys = []
    for i in range(len(x1s)):
        utilitys.append(utility(x1s[i],x2s[i],x3s[i],betas[i,0],betas[i,1],betas[i,2], gamma))
    
    hist, edges = np.histogram(utilitys, bins=150)
    plot1 = plot_hist([hist], [edges],names= [''],plot_range=plot_range,
            y_label='Observations',x_label='Utility',
            title=f'Randomly distributed endowments, gamma = {gamma:.2f}',
            width=500,height=350)
    
    mean = np.mean(utilitys)
    variance = np.var(utilitys)

    meantext = Label(x=250, y=215, text=f'Mean       = {mean:.4f}',
              text_font_size='10pt',x_units='screen', y_units='screen')
    vartext = Label(x=250, y=200, text=f'Variance  = {variance:.4f}',
              text_font_size='10pt',x_units='screen', y_units='screen')
    plot1.add_layout(meantext)
    plot1.add_layout(vartext)
    
    # Equal endowments
    utilitys_equal = []
    for i in range(len(x1s_equal)):
        utilitys_equal.append(utility(x1s_equal[i],x2s_equal[i],x3s_equal[i],betas[i,0],betas[i,1],betas[i,2], gamma))

    
    hist, edges = np.histogram(utilitys_equal, bins=150)
    plot2 = plot_hist([hist], [edges],names= [''],plot_range=plot_range,
                y_label='Observations',x_label='Utility',
                title=f'Equally distributed endowments, gamma = {gamma:.2f}',
                width=500,height=350)
    
    mean = np.mean(utilitys_equal)
    variance = np.var(utilitys_equal)

    meantext = Label(x=250, y=215, text=f'Mean       = {mean:.4f}',
              text_font_size='10pt',x_units='screen', y_units='screen')
    vartext = Label(x=250, y=200, text=f'Variance  = {variance:.4f}',
              text_font_size='10pt',x_units='screen', y_units='screen')
    plot2.add_layout(meantext)
    plot2.add_layout(vartext)
    
    
    return plot1, plot2




'''