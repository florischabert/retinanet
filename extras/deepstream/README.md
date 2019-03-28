# RetinaNet with DeepStream

## Build

* From `extras/deepstream`:
```bash
mkdir build && cd build
cmake -DDeepStream_DIR=/workspace/DeepStream_Release ..
make
```

## Run `deepstream-app`

* From `extras/deepstream`:
```bash
LD_PRELOAD=/retinanet/extras/deepstream/build/libnvdsparsebbox_retinanet.so deepstream-app -c deepstream_app_config_retinanet.txt
```