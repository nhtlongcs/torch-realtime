# Build C++ Code
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=path_to_libtorch ..
make -j4
```

# Convert pytorch model to TorchScript
```
cd ..
# Download baseline.pth and put it here
python convert.py
python inference.py

# There would be baseline_jit.pth file
```

```
cd build
./run ../baseline_jit.pth ../test.jpg # try predicting one image
```
