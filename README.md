# Framing RNN as a kernel method: A neural ODE approach

## Requirements

For reproductibility purposes, we advise to install the project in a dedicated virtual environment to make sure the specific requirements are satisfied.

To install requirements:

```
pip install -r requirements.txt
```

In addition, this package requires installing signatory, which is not entirely straightforward. The following command should work on Linux:

```
pip install signatory==1.2.4.1.7.1 --no-cache-dir --force-reinstall
```

Note that the default MacOS C++ compiler does not support openmp, which is required to compile signatory. A solution is to install the llvm compiler, then set the environment variables so that pip uses this compiler. The following snippet should work.

```
brew install llvm libomp
export CXX=/usr/local/opt/llvm/bin/clang
export LD=/usr/local/opt/llvm/bin/clang
export CC=/usr/local/opt/llvm/bin/clang
pip install signatory==1.2.4.1.7.1 --no-cache-dir --force-reinstall
```

For more detail, see [their documentation](https://signatory.readthedocs.io/en/latest/pages/usage/installation.html).

## Testing

```
python main.py --taylor-exp test
python main.py --adversarial-exp test
```


## Reproducing the paper figures


```
python main.py --taylor-exp final
python main.py --adversarial-exp spirals_penalization
```