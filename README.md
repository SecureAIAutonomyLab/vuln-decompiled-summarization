# Python Package Template
You may use this repository as a template when creating a Python package for your research project. **Please be sure to rewrite this README with relevant content after setting up your package.**

To set up your package, you will need to modify the following:
 - Rename the directory `your_package` to the name of the package you are creating.
 - In `pyproject.toml`:
   - Modify the `project.name` field to be the name of your package.
   - Modify the `project.description` field to be a accurate description of your package.
   - Modify the `project.authors` field to contain all contributor names and emails.
   - Modify the `project.urls."Homepage"` to be the home page of your GitHub repository.
   - Modify the `project.urls."Bug Tracker"` to be the issue page of your GitHub repository.

## Testing Your Code
To test your code without building the package, you will first need `src` to be your current working directory:
```
cd src
```

And then you can execute your Python package  as a runnable module (i.e. the code inside of `__main__.py`):
```
python -m your_packagea arg1 arg2
```
(Replacing `your_package` with the name of your package.)

## Installing Your Code as a Package
To install your code as a package in another project, you may install with `pip`:
```bash
pip install git+https://github.com/SecureAIAutonomyLab/your_package.git#egg=your_package
```
(Replacing `your_package` with the name of your package.)

OR

```bash
git clone https://github.com/SecureAIAutonomyLab/python-package.git
cd python-package
pip install .
```
(If you are using conda to manage your dependencies, you would first need to conda install them via the `environment.yml` file. But if all of your dependecies are specified in your `pyproject.toml`, then you can install directly from github)


## Adding Dependencies
As you are adding additional packages to your project (e.g. `torch`,) please be sure to add them in `environment.yml` and the `dependencies` section of `pyproject.toml`. (Although, since we are mostly going to be using conda for managing our dependencies, you really only need to add them to the `environment.yml` file)