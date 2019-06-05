from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension, CppExtension
import nvidia.dali.sysconfig as dali_sysconfig

ext_modules = []
cxx_args = ['-std=c++11', '-O2', '-Wall']
nvcc_args = [
    '-std=c++11', '--expt-extended-lambda', '--use_fast_math', '-Xcompiler', '-Wall',
    '-gencode=arch=compute_60,code=sm_60', '-gencode=arch=compute_61,code=sm_61',
    '-gencode=arch=compute_70,code=sm_70', '-gencode=arch=compute_72,code=sm_72',
    '-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_75,code=compute_75'
]

# Build torch extensions
ext_modules.append(CUDAExtension('retinanet._C',
    ['csrc/extensions.cpp', 'csrc/tensorrt/engine.cpp', 'csrc/cuda/decode.cu', 'csrc/cuda/nms.cu'],
    extra_compile_args={ 'cxx': cxx_args, 'nvcc': nvcc_args },
    libraries=['nvinfer', 'nvinfer_plugin', 'nvonnxparser'],
))

# Build tensorrt plugins library
ext_modules.append(CUDAExtension('retinanet.tensorrt_plugins',
    ['csrc/tensorrt/plugins.cpp', 'csrc/cuda/decode.cu', 'csrc/cuda/nms.cu'],
    extra_compile_args={ 'cxx': cxx_args, 'nvcc': nvcc_args },
    libraries=['nvinfer', 'nvinfer_plugin', 'nvonnxparser'],
))

# Build dali custom operators library
ext_modules.append(CppExtension('retinanet.dali_operators',
    ['csrc/dali/coco_custom_reader.cpp', 'csrc/dali/json11.cpp'],
    extra_compile_args=cxx_args + dali_sysconfig.get_compile_flags(),
    include_dirs=['/usr/local/cuda/include'],
    extra_link_args=dali_sysconfig.get_link_flags(),
))

setup(
    name='retinanet',
    version='0.1',
    description='Fast and accurate single shot object detector',
    author = 'NVIDIA Corporation',
    author_email='fchabert@nvidia.com',
    packages=['retinanet', 'retinanet.backbones'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=[
        'torch>=1.0.0a0',
        'torchvision',
        'apex @ git+https://github.com/NVIDIA/apex',
        'pycocotools @ git+https://github.com/nvidia/cocoapi.git#subdirectory=PythonAPI',
        'pillow',
        'requests',
    ],
    entry_points = {'console_scripts': ['retinanet=retinanet.main:main']}
)
