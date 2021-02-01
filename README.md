
Install python and pip

pip install -r requirements.txt

Execute python extract_terms.py

## Installation

### Error: the required version of setuptools is not available

Upon running `pip install` or the `setup.py` script you might get a message like this:

```
The required version of setuptools (>=0.7) is not available, and can't be installed while this script is running. Please install a more recent version first, using 'easy_install -U setuptools'.
```

This is appearing because you have a very outdated version of the setuptools package. The redditnlp package typically bootstraps a newer version of setuptools during install, but it isn't working in this case. You need to update setuptools using `easy_install -U setuptools` (you may need to apply `sudo` to this command).

If the above command doesn't do anything then it is likely that your version of setuptools was installed using a package manager such yum, apt or pip. Check your package manager for a package called python-setuptools or try `pip install setuptools --upgrade` and then re-run the install.
