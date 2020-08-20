from numpy                import array, sum, inf, square, mean, sqrt, var, logspace, log10
from tkinter              import Tk
from matplotlib.pyplot    import figure, tight_layout, legend, yscale, xscale, savefig, scatter, plot, title
from matplotlib           import rcParams, rcParamsDefault
from pandas               import read_csv
from tkinter              import filedialog
from tkinter.simpledialog import askinteger
from warnings             import filterwarnings
from sys                  import exit as sysexit
from scipy                import optimize
from scipy.optimize       import differential_evolution

def endProgram():
    sysexit()

def setupTkinterWindow():
    root = Tk()
    root.withdraw()
    return root
    
def quitTkinterWindow(root):
    root.quit()
    root.destroy()

def askForCVSfile():
    filename = filedialog.askopenfilename(filetypes=[("CSV files", ".csv")])
    return filename

def removeTableOverhead(df, r):
    dfcopy = df.copy()
    for i in range(r):
        dfcopy = dfcopy.drop([i]);
    return dfcopy

def removeColnameAndUnitRowsFromDF(df, r):
    dfcopy = df.copy()
    dfcopy = dfcopy.drop([r]);   # Remove colnames row from dataframe
    dfcopy = dfcopy.drop([r+1]); # Remove units row from dataframe
    return dfcopy

def Text2Table(df):
    dfcopy = df.copy()
    dfcopy = dfcopy[0].str.split(",", expand=True);
    return dfcopy

def updateColumnnames(df, variables, units):
    # Show variable names and units as column names
    dfcopy         = df.copy()
    df_header      = [col + " " + uni for col, uni in zip(variables, units)]
    dfcopy.columns = df_header;
    return dfcopy

def renameColumnNames(df, r):
    dfcopy    = df.copy()
    variables = dfcopy.iloc[0]; # get variable names
    units     = dfcopy.iloc[1]; # get units
    dfcopy    = removeColnameAndUnitRowsFromDF(dfcopy, r)
    dfcopy    = updateColumnnames(dfcopy, variables, units)
    return dfcopy

def getDFfromCSV():
    filename = askForCVSfile()
    print(filename.split("/")[-1])
    
    r = 11; # first 11 rows are the overhead
    df = read_csv(filename, header=None, sep='\n');
    df = removeTableOverhead(df, r)
    df = Text2Table(df);
    df = renameColumnNames(df, r);
    
    return df

def readCSVfile():
    try:
        return getDFfromCSV()
    except:
        input("File search was canceled OR the selected file is not a CSV.\nPress ENTER to close.")
        endProgram()

def initializePlotAndSetPlotSize():
    scale  = 6
    plotFigure = figure(figsize=(3*scale, 2*scale))
    tight_layout()
    return plotFigure

def getCurrentAxesInstance(plotFigure):
    return plotFigure.gca()

def showPlotLegend(ax):
    handles, labels = ax.get_legend_handles_labels();
    lgd = dict(zip(labels, handles))
    legend(lgd.values(), lgd.keys(), prop={'size': 22}, loc="best")
    
def setPlotTitle(GNFmodel):
    title(str(GNFmodel).split(" ")[1], size=24);

def namePlotAxes(ax):
    ax.set_xlabel("shear rate " + r"$\frac{1}{s}$", fontsize=24)
    ax.set_ylabel("viscosity " + r"$Pa \cdot s$", fontsize=24)

def formatTicksAndLabelFontSizes(ax):
    for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(18)
    ax.tick_params(which='both', direction='in', length=5, width=2, bottom=True, top=True, left=True, right=True)
    
def setLogScale():
    yscale('log')
    xscale('log')
    
def savePlot(GNFmodel):
    savefig(
        str(GNFmodel).split(" ")[1] + 'fit.png',
        dpi=200,
        bbox_inches='tight')
    
def recoverMatplotlibDefaults():
    rcParams.update(rcParamsDefault)

def getColumnNames(df):
    x_str = df.columns[0]
    y_str = df.columns[1]
    return x_str, y_str

def removeNANsFromData(df, x_str, y_str):
    df = df.dropna(subset=[x_str, y_str])
    return df
    
def getXYdata(df):
    x_str, y_str = "Shear rate 1/s", "Viscosity Pa.s"
    df           = removeNANsFromData(df, x_str, y_str)
    x            = df[x_str].astype(float);
    y            = df[y_str].astype(float);
    x, y         = pandasSeries2numpyArray(x, y);
    return x, y

def scatterExperimentalData(x, y):
    scatter(x, y, s=20, marker='o', label="data")
    #plot(x, y, linewidth=1, linestyle='-.')

def Cross_(gamma, *p):
    eta_zero = p[0];
    eta_inft = p[1];
    kappa    = p[2];
    n        = p[3];
    a        = p[4];
    
    nume = eta_zero - eta_inft;
    deno = 1 + (kappa * gamma)**n;
    return (nume/deno) + eta_inft;

def Carreau_(gamma, *p):
    eta_zero = p[0];
    eta_inft = p[1];
    kappa    = p[2];
    n        = p[3];
    a        = p[4];
    
    nume = eta_zero - eta_inft;
    base = 1 + (kappa * gamma)**2;
    expo = (1 - n)/2;
    deno = base**expo;
    return (nume/deno) + eta_inft;

def CarreauYasuda_(gamma, *p):
    eta_zero = p[0];
    eta_inft = p[1];
    kappa    = p[2];
    n        = p[3];
    a        = p[4];
    
    nume = eta_zero - eta_inft;
    base = 1 + (kappa * gamma)**a;
    expo = (1 - n)/a;
    deno = base**expo;
    return (nume/deno) + eta_inft;

def Sisko_(gamma, *p):
    eta_zero = p[0];
    eta_inft = p[1];
    kappa    = p[2];
    n        = p[3];
    a        = p[4];
    
    expo = n - 1;
    return eta_inft + kappa * (gamma**expo);

def Williamson_(gamma, *p):
    eta_zero = p[0];
    eta_inft = p[1];
    kappa    = p[2];
    n        = p[3];
    a        = p[4];
    
    deno = 1 + (kappa * gamma)**n;
    return eta_zero/deno;

def generatePlot(x, y, xModel, yModel, GNFmodel):
    plotFigure = initializePlotAndSetPlotSize()
    ax0        = getCurrentAxesInstance(plotFigure)
    scatterExperimentalData(x, y)
    plotFittedCurve(xModel, yModel)
    namePlotAxes(ax0)
    formatTicksAndLabelFontSizes(ax0)
    setLogScale()
    showPlotLegend(ax0)
    setPlotTitle(GNFmodel)
    savePlot(GNFmodel)
    recoverMatplotlibDefaults()
        
def pandasSeries2numpyArray(x, y):
    x = array(x)
    y = array(y)
    return x, y

def getMaxValue(y):
    minY  = abs(min(y))
    maxY  = abs(max(y))
    return minY, maxY

def generate_Initial_Parameters(x, y, GNFmodel, parameterBounds):
    # function for genetic algorithm to minimize (sum of squared error)
    def sumOfSquaredError(parameterTuple):
        filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = GNFmodel(x, *parameterTuple)
        return sum((y - val) ** 2.0)
    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

def getParameterBounds(x, y):
    # min and max used for bounds
    minY, maxY = getMaxValue(y)
    parameterBounds = []
    parameterBounds.append([0, maxY])       # eta_zero
    parameterBounds.append([minY/10, minY]) # eta_inft
    parameterBounds.append([0, 500])        # kappa
    parameterBounds.append([-1000, 10])     # n
    parameterBounds.append([0, 1000])       # a
    return parameterBounds

def explodeParameterBoundsInto2Arrays(parameterBounds):
    lowbound=[]
    upbound=[]
    for bound in parameterBounds:
        lowbound.append(bound[0])
        upbound.append(bound[1])
    return lowbound, upbound

def computeStatistics(x, y, fittedParameters, GNFmodel):
    modelPredictions = GNFmodel(x, *fittedParameters) 
    absError = modelPredictions - y
    SE = square(absError) # squared errors
    MSE = mean(SE) # mean squared errors
    RMSE = sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (var(absError) / var(y))
    return RMSE, Rsquared
    
def curveFitTheTestData(GNFmodel, x, y, geneticParameters, lowbound, upbound):
    try:
        fittedParameters, pcov = optimize.curve_fit(
            GNFmodel, x, y, geneticParameters,
            bounds=(lowbound, upbound));
        return fittedParameters
    except:
        print("Optimal parameters not found: The maximum number of function evaluations is exceeded.\nTry a different number of Maxwell elements (or add more data points).")
        
def printStatistics(fittedParameters, RMSE, Rsquared):
    print('[eta_zero, eta_inft, kappa, n, a]\n', fittedParameters)
    print('Root Mean Squared Error :', RMSE)
    print('R-squared               :', Rsquared)
    
def calculateXYpredictions(x, GNFmodel, fittedParameters):
    minxMagnitude = log10(min(x))
    maxxMagnitude = log10(max(x))
    xModel = logspace(minxMagnitude, maxxMagnitude)
    yModel = GNFmodel(xModel, *fittedParameters)
    return xModel, yModel, GNFmodel
    
def fitdata(x, y, GNFmodel):
    parameterBounds   = getParameterBounds(x, y)
    geneticParameters = generate_Initial_Parameters(x, y, GNFmodel, parameterBounds)
    lowbound, upbound = explodeParameterBoundsInto2Arrays(parameterBounds)
    fittedParameters  = curveFitTheTestData(GNFmodel, x, y, geneticParameters, lowbound, upbound)
    RMSE, Rsquared    = computeStatistics(x, y, fittedParameters, GNFmodel)
    printStatistics(fittedParameters, RMSE, Rsquared)
    return calculateXYpredictions(x, GNFmodel, fittedParameters)

def plotFittedCurve(xModel, yModel):
    plot(xModel, yModel, linestyle='-', linewidth=4, label="fit")
    
def fitExperimentalData(GNFmodel):
    root     = setupTkinterWindow()
    df       = readCSVfile()
    x, y     = getXYdata(df)
    model    = fitdata(x, y, GNFmodel)
    generatePlot(x, y, model[0], model[1], model[2])
    quitTkinterWindow(root)
    print()