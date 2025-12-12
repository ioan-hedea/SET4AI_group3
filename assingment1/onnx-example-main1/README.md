# CS4015 Software engineering and testing for AI systems

This folder contains the example for converting your model to onnx, using the onnx runtime.

#
### <u>Pipenv</u>

To make life easy for everyone, we've setup a pip file to ensure quick and easy install of dependencies.

<b>NOTE</b>: <i>Before installing any dependencies, open the Pipfile in your editor and un-comment the version of Tensorflow for your system.</i>

Open a terminal/ powershell and navigate to the project folder, then type:

    pipenv shell

This will put your current session into the python virtual environment. Then type:

    pipenv install

This will install the dependencies defined in the Pipfile, into this specific environment. By doing this, we can ensure no cross dependency issues when working on different python projects.

---

You will need to enable this virtual environment in your code editor to ensure it uses the correct dependencies. For VS Code, this can be found in the bottom right corner of the UI.

It will currently likely show your current Python version. Click this and it will open up the 'Select Interpreter' drop down. For myself, the environment starts with <b><i>'labs'</i></b>, which I then click on to enable as my interpreter.

Yours will likely be the same, or if different, will be shown in your terminal/ powershell window when you typed 'pipenv shell' before.

That should have you up and running! Enjoy the labs and if you have any issues with this, please reach out to the staff and we'll do our best to get you going.