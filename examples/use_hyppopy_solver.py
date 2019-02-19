import os
import sys
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# the ProjectManager is loading your config file and giving you access
# to everything specified in the settings/custom section of the config
from hyppopy.projectmanager import ProjectManager

# the SolverFactory builds the Solver class for you
from hyppopy.solverfactory import SolverFactory

# we use in this example the SimpleDataLoader
from hyppopy.workflows.dataloader.simpleloader import SimpleDataLoader

# until Hyppopy is not fully installable we need
# to set the Hyppopy package folder by hand
HYPPOPY_DIR = "D:/MyPythonModules/hyppopy"
sys.path.append(HYPPOPY_DIR)

# let the ProjectManager read your config file
DATA = os.path.join(HYPPOPY_DIR, *("hyppopy", "tests", "data", "Titanic"))
ProjectManager.read_config(os.path.join(DATA, 'rf_config.json'))

# ----- reading data somehow ------
dl = SimpleDataLoader()
dl.start(path=ProjectManager.data_path,
         data_name=ProjectManager.data_name,
         labels_name=ProjectManager.labels_name)
# ---------------------------------

# ----- defining loss function ------
def blackbox_function(params):
    clf = SVC(**params)
    return -cross_val_score(estimator=clf, X=dl.data[0], y=dl.data[1], cv=3).mean()
# -----------------------------------

# ----- create and run the solver ------
# get a solver instance from the SolverFactory
solver = SolverFactory.get_solver()
# set your loss function
solver.set_loss_function(blackbox_function)
# run the solver
solver.run()
# store your results
solver.save_results(savedir="C:\\Users\\Me\\Desktop\\myTestProject")
# --------------------------------------