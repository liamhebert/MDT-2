# MDT-2 Rewrite

Work-in-progress rewrite of MDT, good to look at for how the Hydra+Lightning template can be used. 

## Setting up formatter rules

Add these lines to your settings.json for vscode. You can access your settings
by typing

`> preferences: Open user settings (JSON)`

then just paste these anywhere (assuming you don't already have values for
these).

You might also need the black-formatter extension, but I believe it comes
pre-installed with the python extension.

```
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnPaste": true,
    "editor.formatOnSave": true,
    "editor.rulers": [
        80
    ],
    "black-formatter.args": [
        "--line-length",
        "80"
    ],
```

## Setting up auto-run pre-commit rules

Run `pre-commit install` in the root folder of the repo. This will run all the
pre-commit fixes before you commit code.

You can also run them manually by running `pre-commit` in the root folder.
