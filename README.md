# Firewatch: A MICRO-452 Project

Using the Thymio Robot within Aseba Studio, and simulating an environment wherein the goal is to put out a fire as fast as possible. This involves vision to detect the fire and obstacles, global navigation and path planning to determine best path to the fire, pose estimation to correct and adapt robot path while accounting for uncertainties in its sensors and motion, local navigation to avoid obstacles that may appear suddenly (rocks or tree debris) and controllers to make robot follow a given path.

<!-- Badge URLs point at this repository; update the owner/repo path after generating a new repository from this template -->
[![CI](https://github.com/sohamm29/epfl-mobile-robotics-project/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sohamm29/epfl-mobile-robotics-project/actions/workflows/ci.yml)

</div>

---

## Project Structure

```text
.
├── .github/                            # contains CI workflows for maintaining code quality (linting)
├── assets/                             # contains all images/artifacts that are useful across project
├── docs/                               # comprehensive documentation should go here
├── sandbox/                            # test notebooks, scripts etc to test things
├── src/                                # main code folder and report location
    └── report.ipynb                    
├── .editorconfig                       # ensures VSCode has same setup for everyone
├── .gitattributes                      # ensures all line endings are same (avoid different OS issues)
├── .gitconfig                          # useful git aliases to use
├── .gitignore                          # tells git which files not to send to Github (cache, venv)
├── .pre-commit-config.yaml             # code quality enforcer, makes sure correct formatting and so on
├── .python-version                     # tells system which python version to use
├── ci.sh                               # run code quality tests locally, to ensure all good
├── pyproject.toml                      # ensures project dependencies and libraries
└── README.md                           # this document

```

## Maintainers

[Soham Malhotra](https://github.com/sohamm29)
[Paul Lefeuvre](https://github.com/PaulLefeuvre)
[Remi Khani](https://github.com/sohamm29)
[Sébastien Pegada](https://github.com/sohamm29)
