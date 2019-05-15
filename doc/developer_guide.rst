****************
Developers Guide
****************

The main classes and their connections
**************************************

The picture below depicts the releationships between the most important classes of hyppopy.

.. image:: _static/class_diagram.png

To understand the concept behind Hyppopy the following classes are important:
 - :py:mod:`hyppopy.solvers.HyppopySolver`
 - :py:mod:`hyppopy.HyppopyProject` 
 - :py:mod:`hyppopy.BlackboxFunction`

 
The :py:mod:`hyppopy.solvers.HyppopySolver` class is the parent class of all solvers in Hyppopy. It defines 
an abstract interface that needs to be implemented by each custom solver class. The main idea is to
define a common interface for the different approaches the solver libraries are based on. When designing
Hyppopy there were three main challenges that drove the design. Each solver library has a different 
approach to define or describe the hyperparameter space, has a different approach to track the solver
information and is different in setting the blackbox function and running the optimization process. To
deal with those differences the :py:mod:`hyppopy.solvers.HyppopySolver` class defines the abstract interface
functions `convert_searchspace`, `execute_solver`, `loss_function_call` and `define_interface`. Those serve as 
abstraction layer to handle the individual needs of each solver library. 

Each solver needs a :py:mod:`hyppopy.HyppopyProject` instance keeping the user configuration input and a 
:py:mod:`hyppopy.BlackboxFunction` instance, implementing the loss function.

Implementing a custom solver
****************************

Adding a new solver is only about deriving a new class from :py:mod:`hyppopy.solvers.HyppopySolver` as well as
telling the :py:mod:`hyppopy.SolverPool` that it exists. We go through the whole process on the example of the 
solver :py:mod:`hyppopy.solvers.OptunitySolver`:

.. code-block:: python

	import os
	import optunity
	from pprint import pformat


	from hyppopy.solvers.HyppopySolver import HyppopySolver


	class OptunitySolver(HyppopySolver):

		def __init__(self, project=None):
			HyppopySolver.__init__(self, project)

First step is to derive from the HyppopySolver class. Good practice would be that the project can be set via __init__
and if, is piped through to the HyppopySolver.__init__.

Next step is implementing the abstract interface methods. We start with define_interface. This functions purpose is to
define the relevant input parameter and the signature of a hyperparameter description. This means the solver developer
can define what parameter the solver expects as well as how a single hyperparameter must be described. The rules defined
here are automatically applied when the solver run method is called and exceptions are thrown if there is a mismatch
between these rules and the settings the user sets via it's config.

Our solver in this example needs an parameter called max_iterations of type int. The hyperparameter space has a domain
that allows values 'uniform' and 'categorical', a field data of type list and a field type of type type. This guarantees
that exceptions are thrown if the user disrespects this signature or forgets to set max_iterations.

.. code-block:: python

    def define_interface(self):
        self._add_member("max_iterations", int)
        self._add_hyperparameter_signature(name="domain", dtype=str,
                                          options=["uniform", "categorical"])
        self._add_hyperparameter_signature(name="data", dtype=list)
        self._add_hyperparameter_signature(name="type", dtype=type)
		
	
Next abstract method to implement is convert_searchspace. This method is responsible for interpreting the users hyperparameter
input and convert it to a form the solver framework needs. An input for example can be:

.. code-block:: python

	hyperparameter = {
		'C': {'domain': 'uniform', 'data': [0.0001, 20], 'type': float},
		'gamma': {'domain': 'uniform', 'data': [0.0001, 20.0], 'type': float},
		'kernel': {'domain': 'categorical', 'data': ['linear', 'sigmoid', 'poly', 'rbf'], 'type': str},
		'decision_function_shape': {'domain': 'categorical', 'data': ['ovo', 'ovr'], 'type': str'}
	}


Optunity instead expects a hyperparameter space formulation as follows:
 
.. code-block:: python

	optunity_space = {'decision_function_shape': 
	{'ovo': {
		'kernel': {
			'linear': {'C': [0.0001, 20], 'gamma': [0.0001, 20.0]},
			'sigmoid': {'C': [0.0001, 20], 'gamma': [0.0001, 20.0]},
			'poly': {'C': [0.0001, 20], 'gamma': [0.0001, 20.0]},
			'rbf': {'C': [0.0001, 20], 'gamma': [0.0001, 20.0]}}
		}, 
	'ovr': {
		'kernel': {
			'linear': {'C': [0.0001, 20], 'gamma': [0.0001, 20.0]},
			'sigmoid': {'C': [0.0001, 20], 'gamma': [0.0001, 20.0]},
			'poly': {'C': [0.0001, 20], 'gamma': [0.0001, 20.0]},
			'rbf': {'C': [0.0001, 20], 'gamma': [0.0001, 20.0]}}
		}
	}}
				 
This conversion is what convert_searchspace is meant for. 

.. code-block:: python

	def convert_searchspace(self, hyperparameter):
        LOG.debug("convert input parameter\n\n\t{}\n".format(pformat(hyperparameter)))
        # split input in categorical and non-categorical data
        cat, uni = self.split_categorical(hyperparameter)
        # build up dictionary keeping all non-categorical data
        uniforms = {}
        for key, value in uni.items():
            for key2, value2 in value.items():
                if key2 == 'data':
                    if len(value2) == 3:
                        uniforms[key] = value2[0:2]
                    elif len(value2) == 2:
                        uniforms[key] = value2
                    else:
                        raise AssertionError("precondition violation, optunity searchspace needs list with left and right range bounds!")

        if len(cat) == 0:
            return uniforms
        # build nested categorical structure
        inner_level = uniforms
        for key, value in cat.items():
            tmp = {}
            optunity_space = {}
            for key2, value2 in value.items():
                if key2 == 'data':
                    for elem in value2:
                        tmp[elem] = inner_level
            optunity_space[key] = tmp
            inner_level = optunity_space
        return optunity_space
		
		
Now we have defined how the solver looks from outside and how to convert the parameterspace coming in, we can define how the blackbox function
is called. The abstract method loss_function_call is a wrapper function enabling to customize the call of the blackbox function. In case of Optunity
we only check if a parameter is of type int and convert it to ensure that no exception are thrown in case of integers are expected in the blackbox.

.. code-block:: python

	def loss_function_call(self, params):
        for key in params.keys():
            if self.project.get_typeof(key) is int:
                params[key] = int(round(params[key]))
        return self.blackbox(**params)
		

In execute_solver the actual wrapping of the solver framework call is done. Here call the Optunity optimizing function. A dictionary keeping the optimal 
parameter set must assigned to self.best.   


.. code-block:: python

	def execute_solver(self, searchspace):
        LOG.debug("execute_solver using solution space:\n\n\t{}\n".format(pformat(searchspace)))
        try:
            self.best, _, _ = optunity.minimize_structured(f=self.loss_function,
                                                           num_evals=self.max_iterations,
                                                           search_space=searchspace)
        except Exception as e:
            LOG.error("internal error in optunity.minimize_structured occured. {}".format(e))
            raise BrokenPipeError("internal error in optunity.minimize_structured occured. {}".format(e))
			

