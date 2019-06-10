nvcc=/home/dongwool/usr/local/cuda-10.0/bin/nvcc
cudainc=/home/dongwool/usr/local/cuda-10.0/include/
cudalib=/home/dongwool/usr/local/cuda-10.0/lib64/
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


$nvcc tf_interpolate_gpu.cu -o tf_interpolate_gpu.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_interpolate_gpu.cpp tf_interpolate_gpu.cu.o -o tf_interpolate_gpu_so.so -shared -fPIC -I $TF_INC -I $cudainc -I$TF_INC/external/nsync/public -lcudart -L $cudalib -L$TF_LIB -ltensorflow_framework -O2
