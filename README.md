# QLearning
A minimalist python library, built in C++, for Tabula QLearning.

## Using the library
The libary can be either built using cmake from the project directory or it can be downloaded from github actions.
For examples of usage go to `./python/`.

### Downloading from artifacts
Every push to main *should* (if it doesn't open an issue please :)) run the CI which uploads a built library as a github actions artifact!
To find this:
* Go to the **Actions** tab of the repository
* Click on the most recent succesful run
* Go to the build section and expand the section named **Upload library**
* Download a zip containing the libary from there!

### Building
To build, get all the dependencies using nix flakes
```bash
nix develop
```
and then make the build repository
```bash
cmake -B ./build
```
From here we want to actually build the library which we do with
```bash
cmake --build ./build/
```
