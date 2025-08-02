### UV setup
If `uv` is not installed, install it. <br>
https://docs.astral.sh/uv/ <br>
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install
```
### Make environment
make environment with
```
make install
```
### Folder system
`.trash` throw all your junk here <br>
`.venv` python virtual environment <br>
`exp` experimental folders <br>
`ref` referential materials <br>
`shared` folder you symlink to your projects <br>

### Symlink shared folder
```
ln ../../shared -s
```

### General library usage
`pyyaml` to store configs <br>
`tensorboard` to visualize training runs <br>
