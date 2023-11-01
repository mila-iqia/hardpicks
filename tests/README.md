Tests
-----

This folder contains the test suite for the project.

It is often necessary to "mock" certain elements in the main code in order to test effectively.
Here we'll list useful tricks we use to work around common difficulties.


* avoiding mlflow logging

    the 'mock.patch' construct will simply "turn off" the function referred to in the argument.

>   with mock.patch("hardpicks.utils.hp_utils.log_hp"):
> 
>      [some code that now won't try to log]
 

* substitute one 'real' internal object for a 'fake' object more relevant for the purpose of testing

>   with mock.patch.object([object module path], [object name], new=[new object]):
> 
>       [some code that will now use the 'new' object
