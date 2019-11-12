### Introduction

We will use forward mode to solve differentiation problems for users. It’s important because sometimes differentiation requires tedious calculations, and we want computers to solve these problems automatically for us.

### Background

Chain rule and graph structures will be the cornerstone underlying our program to differentiate various functions. The differentiation process of a complex function can be decomposed into calculating the derivatives of a chain of simple functions. The idea of computation graph will guide how the chain of derivatives will be evaluated and combined to reach the final answer.

### How to Use PackageName

The user will import our package and have two ways to use it. The first way is similar to the AutoDiffToy example. The user initializes a base object and is then able to build their function up by adding, multiplying, etc. other constants or objects.

In addition the user can use command-line input to enter the functions they want to evaluate and at what values. If the output from the function is a vector, the user can enter these and store them in a list, as well as the value or values of alpha as a scalar or vector that they want to evaluate the function at. The user will pass these to the AD object which will evaluate the function and the derivative at the given values of alpha and store these, which the user can retrieve.

To evaluate a vector functions, the mathematical functions will be passed into our program as a list of functions. Our program will evaluate these functions one by one. To evaluate scalar functions of vectors, the user will first input the vector of values they want to evaluate at. Based on the length of the vector, our program could determine how many variables there are in the function and use this information to parse the input function.

A couple examples are shown below with pseudocode. The first is similar to the AutoDiffToy example and would probably be how a user would use our package in their program. The second is more of a demo of how the package works, and the user can enter functions using command-line input and pass them to the demo class.

```
import AD

x = AD.AutoDiffForward(2.0)
f_x = 3 * x ** 2 + 2 * x + 5
print(f_x.val, f_x.der)

y1 = AD.AutoDiffForward(3.0)
y2 = AD.AutoDiffForward(4.0)
f_y = 3 * y1 **2 + 2 * y1 * AD.sin(y2) + 5
print(f_y.val, f_y.der)

f = input(‘Input the function’)
alpha = input(‘Input the value to evaluate the function and derivative’)
ad = AD.AutoDiffForwardDemo([alpha], [f])

print(ad.val, ad.der)
```

### Software Organization

Discuss how you plan on organizing your software package.

* What will the directory structure look like?

```
dir\
	driver.py
		AutoDifferentiation\
			__init__.py
			main\
				__init__.py
				AutoDiffForward.py
				input_parsing.py
				func_eval.py
				user_interface.py
			util\
				eval_rules_for_simple_fun.py
			test\
				__init__.py
				test_main.py
				test_eval.py
				test_user_interface.py
```

* What modules do you plan on including? What is their basic functionality?  
We will have a main module, AutoDiffForward, for the user to use the forward mode of auto-differentiation of our package. We will also have a few utility packages, like input_parsing and fun_eval which help us keep our code organized by breaking the different related functions into modules. We will also have several modules for testing our package.
    

* Where will your test suite live? Will you use TravisCI? CodeCov?  
The test suites will live in test directory. Yes, we will use both TravisCI and CodeCov for testing.

* How will you distribute your package (e.g. PyPI)?  
We will distribute our package using TestPyPI.

* How will you package your software? Will you use a framework? If so, which one and why? If not, why not?  
We will use PyScaffold to package our software. We choose it because it is convenient.

### Implementation

* What are the core data structures?  
We will store vector input and output of functions as lists. We shouldn't need any more advanced data structures than lists for our program.

* What classes will you implement?  
The main class we will implement for forward mode is `AutoDiffForward`, which will be the main interface for the user to use our package. We will also implement `AutoDiffForwardDemo` which can accept string representation of functions.

* What method and name attributes will your classes have?  
Attributes: `val`, `der` for the value and derivative of the function. If the function output is a vector, these will be lists.  
We will implement `__add__`, `__raad__`, `__mul__`, `__rmul__`, `__sub__`, `__rsub__`, `__truediv__`, `__rtruediv__`, `__pow__`, `sin`, `tan`, `cos`, `exp`, `log`, `sqrt` for the AutoDiffForward class.

* What external dependencies will you rely on?  
Currently we only plan to use numpy.

* How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?  
We will use numpy’s functions to handle the function value of these elementary functions. We will have python functions or classes to evaluate the derivatives of each of these, so our package knows the derivative rules for each elementary function.
